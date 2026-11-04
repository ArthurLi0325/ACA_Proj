#!/usr/bin/env python3
"""Export one DeiT-Tiny encoder block to C headers plus golden tensors.

usage: export_block.py model.tflite [--block 0] [--input x.npy|synthetic] [--out build/model]
                       [--golden build/golden] [--no-check]
       export_block.py model.tflite --range FIRST:LAST --prefix NAME [--base N] ...   (stem / tail,
                       used by export_model.py)

Writes
  <out>/block<b>_params.h   weights, folded biases, fixed-point multipliers, zero points
  <out>/block<b>_golden.h   block input and expected block output (int8)
  <out>/block<b>_golden_<op>.h  input(s) and expected output of each FC / BATCH_MATMUL
                            (qkv, proj, fc1, fc2, qk, av) for kernel-level tests
  <golden>/block<b>/        every tensor the block reads or writes as .npy + manifest.json

Zero-point folding (DESIGN.md D2): weights are symmetric (zp = 0), so for FC
  sum_k (a_k - zp_a) w_k + b  =  sum_k a_k w_k + (b - zp_a * sum_k w_k)
and the parenthesised term is stored as the folded int32 bias. BATCH_MATMUL has
activations on both sides, so its zero points cannot be folded offline.

Fixed-point multipliers follow TFLite QuantizeMultiplier: real = M0 * 2^(shift - 31),
M0 in [2^30, 2^31) (MEAN: smaller, see mean_multiplier), shift > 0 means left shift. A block starts at the MEAN of its
first LayerNorm and ends at the residual ADD before the next block's LayerNorm.
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---- TFLite fixed-point helpers (gemmlowp semantics, double rounding) ----

def quantize_multiplier(real):
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    q_fixed = int(math.floor(abs(q) * (1 << 31) + 0.5)) * (1 if q > 0 else -1)  # round half away
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q_fixed, shift


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul on int64 arrays holding int32 values."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    s = ab + nudge
    res = np.where(s >= 0, s >> 31, -((-s) >> 31))          # C division truncates toward 0
    return np.where((a == INT32_MIN) & (b == INT32_MIN), INT32_MAX, res)


def rounding_divide_by_pot(x, e):
    x = np.asarray(x, np.int64)
    e = np.asarray(e, np.int64)
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def mbqm(x, m0, shift):
    """MultiplyByQuantizedMultiplier(x, M0, shift)."""
    shift = np.asarray(shift, np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rounding_divide_by_pot(srdhm(np.asarray(x, np.int64) << left, m0), right)


def mbqm_halfup(x, m0, shift):
    """Variant whose final right shift rounds ties toward +inf instead of away from 0.
    LiteRT's x86 FULLY_CONNECTED path (ruy) matches this one bit-exactly."""
    shift = np.asarray(shift, np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    v = srdhm(np.asarray(x, np.int64) << left, m0)
    return np.where(right > 0, (v + ((np.int64(1) << right) >> 1)) >> right, v)


def mean_multiplier(s_in, s_out, n):
    """LiteRT's int8 MEAN multiplier: QuantizeMultiplier(s_in/s_out), then divided by N in
    fixed point (M0 << sh) // N with sh = floor(log2 N). Rounding s_in/(s_out*N) directly gives
    a slightly different M0 and misses ~25% of outputs by 1 in later blocks."""
    m, e = quantize_multiplier(s_in / s_out)
    sh = min(n.bit_length() - 1, 32, 31 + e)
    return (m << sh) // n, e - sh


def mbqm_single(x, m0, shift):
    """MultiplyByQuantizedMultiplier with TFLITE_SINGLE_ROUNDING: one rounding, ties toward +inf.
    LiteRT's x86 int8 ADD/SUB match this one (all 100 ADD/SUB ops of the model), not mbqm."""
    x = np.asarray(x, np.int64)
    ts = 31 - np.asarray(shift, np.int64)
    return (x * np.asarray(m0, np.int64) + (np.int64(1) << (ts - 1))) >> ts


def rq_away(x, m0, shift, zo):
    """clamp(MultiplyByQuantizedMultiplier(x) + zo): what the rq.a instruction computes."""
    return np.clip(mbqm(x, m0, shift) + zo, -128, 127)


def addsub_int8(a, za, sa, b, zb, sb, so, zo, sign):
    """TFLite int8 ADD/SUB: both inputs rescaled to 2*max(sa, sb) with 20 extra bits, every
    rescaling with single rounding (mbqm_single). The C code (src/common/vit.h) folds the output
    multiplier into 64-bit input tables; this is the same arithmetic."""
    twice = 2 * max(sa, sb)
    (ma, ea), (mb, eb) = quantize_multiplier(sa / twice), quantize_multiplier(sb / twice)
    mo, eo = quantize_multiplier(twice / ((1 << 20) * so))
    a = mbqm_single((np.asarray(a, np.int64) - za) << 20, ma, ea)
    b = mbqm_single((np.asarray(b, np.int64) - zb) << 20, mb, eb)
    return np.clip(mbqm_single(a + sign * b, mo, eo) + zo, -128, 127)


# ---- int8 -> int8 lookup tables and the integer softmax table ----

def _srdhm1(a, b):
    return int(srdhm(a, b))


def _sat_shl(x, e):
    return max(INT32_MIN, min(INT32_MAX, x << e))


def _mbqm1(x, m0, shift):
    return int(mbqm(x, m0, shift))


def inv_sqrt_multiplier(v):
    """TFLite GetInvSqrtQuantizedMultiplierExp(v, kReverseShift): gemmlowp Newton-Raphson."""
    if v <= 1:
        return INT32_MAX, 0
    shift = 11
    while v >= (1 << 29):
        v //= 4
        shift += 1
    pairs = (32 - v.bit_length() - 1) // 2 - 1
    shift -= pairs
    v <<= 2 * pairs
    half = int(rounding_divide_by_pot(v >> 1, 1))              # F3 input / 2
    x = 1 << 28                                                 # F3 1.0
    for _ in range(5):
        x3 = _sat_shl(_srdhm1(_srdhm1(x, x), x), 6)             # F9 -> F3
        x = _sat_shl(_srdhm1(3 << 27, x) - _srdhm1(half, x3), 3)  # F6 -> F3
    x = _srdhm1(x, 1518500250)                                  # * sqrt(2)/2
    if shift < 0:
        x <<= -shift
        shift = 0
    return x, -shift


def rsqrt_lut(si, zi, so, zo):
    """LiteRT int8 RSQRT (integer code path) evaluated for every input value."""
    m0, sh = quantize_multiplier(1.0 / (math.sqrt(si) * so))
    lut = np.zeros(256, np.int64)
    for q in range(-128, 128):
        v = q - zi
        if v <= 0:                         # negative inputs are rejected by LiteRT; 0 -> max
            lut[q & 0xFF] = 127
            continue
        im, ish = inv_sqrt_multiplier(v)
        y = _mbqm1(_mbqm1(1, im, ish + 20), m0, sh - 20) + zo
        lut[q & 0xFF] = max(-128, min(127, y))
    return lut


def gelu_lut(si, zi, so, zo, approximate):
    """LiteRT LUTPopulate<int8_t> for GELU, emulated in float32."""
    f32 = np.float32
    inv = f32(1) / f32(so)
    lut = np.zeros(256, np.int64)
    for q in range(-128, 128):
        x = f32(si) * f32(q - zi)
        if approximate:
            inner = f32(f32(math.sqrt(2 / math.pi)) * f32(x + f32(f32(f32(f32(0.044715) * x) * x) * x)))
            y = f32(f32(f32(0.5) * x) * f32(f32(1) + f32(math.tanh(float(inner)))))
        else:
            y = f32(f32(f32(0.5) * x) * f32(math.erfc(float(f32(x * f32(-math.sqrt(0.5)))))))
        r = float(f32(y * inv))
        r = math.floor(abs(r) + 0.5) * (1 if r >= 0 else -1)   # TfLiteRound
        lut[q & 0xFF] = max(-128, min(127, int(f32(f32(r) + f32(zo)))))
    return lut


SOFTMAX_EXP_BITS = 24


def softmax_f32_table(si, beta):
    """LiteRT's float table: T[255 - v] = expf(-si*beta*v), as float32."""
    f32 = np.float32
    scale = f32(-f32(si) * f32(beta))
    t = np.zeros(256, f32)
    for v in range(256):
        t[255 - v] = f32(math.exp(float(f32(scale * f32(v)))))
    return t


def softmax_exp_table(si, beta):
    """E[i] = round(T[i] * 2^24): the float table made integer (T[255] = 1.0 -> 2^24)."""
    return np.round(softmax_f32_table(si, beta).astype(np.float64) * (1 << SOFTMAX_EXP_BITS)).astype(np.int64)


def softmax_row_f32(r, t, so, zo):
    """One row of LiteRT's x86 int8 SOFTMAX exactly: float32 sum in index order, 1/(sum*so),
    std::round(T[x] * inv) + zo. Reproduces all 12 SOFTMAX ops of the model bit-exactly."""
    f32 = np.float32
    e = t[255 - r.max() + r]
    s = f32(0)
    for v in e:
        s = f32(s + v)
    inv = f32(f32(1) / f32(s * f32(so)))
    return np.clip(np.floor((e * inv).astype(f32).astype(np.float64) + 0.5).astype(np.int64) + zo, -128, 127)


def softmax_int(x, si, beta, so, zo):
    """The softmax of src/common/vit.h. Fast path: integer table at 2^24, one division per row,
    q = e*R / 2^46 with R = 2^54 // sum. q differs from LiteRT's float32 result by less than
    q*2^-15 + 2^-16 (float32 sum: < 199 * 2^-24 relative; integer path: < 6.2e-6 relative plus
    7.6e-6 absolute), so when no element is that close to a rounding tie the fast path is exact;
    otherwise the row is recomputed with an integer emulation of LiteRT's float32 arithmetic.
    Returns (output, rows that took the exact path)."""
    x = np.asarray(x, np.int64)
    t = softmax_f32_table(si, beta)
    ti = softmax_exp_table(si, beta)
    out = np.zeros(x.shape, np.int64)
    slow = 0
    for idx in np.ndindex(x.shape[:-1]):
        r = x[idx]
        e = ti[255 - r.max() + r]
        R = (1 << 54) // int(e.sum())
        q = e * R
        qh = q + (1 << 45)
        g = (q >> 15) + (1 << 30)
        if (((qh + g) & ((1 << 46) - 1)) < 2 * g).any():
            out[idx] = softmax_row_f32(r, t, so, zo)
            slow += 1
        else:
            out[idx] = np.minimum((qh >> 46) + zo, 127)
    return out, slow


# ---- graph helpers ----

def builtin_options(model_buf, op_index):
    m = tflite.Model.GetRootAsModel(model_buf, 0)
    op = m.Subgraphs(0).Operators(op_index)
    return op, op.BuiltinOptions()


def bmm_adj(model_buf, op_index):
    op, t = builtin_options(model_buf, op_index)
    o = tflite.BatchMatMulOptions()
    o.Init(t.Bytes, t.Pos)
    return bool(o.AdjX()), bool(o.AdjY())


def qparams(t):
    q = t["quantization_parameters"]
    return [float(s) for s in q["scales"]], [int(z) for z in q["zero_points"]]


def block_range(ops, b):
    means = [op["index"] for op in ops if op["op_name"] == "MEAN"]
    start = means[4 * b]                      # 2 LayerNorms x 2 MEANs per block
    end = means[4 * (b + 1)] - 1 if 4 * (b + 1) < len(means) else len(ops) - 1
    return start, end


# ---- C emission ----

def c_array(ctype, name, arr, per_line=24):
    flat = np.asarray(arr).reshape(-1)
    body = ",\n".join("  " + ", ".join(str(int(v)) for v in flat[i:i + per_line])
                      for i in range(0, len(flat), per_line))
    return f"static const {ctype} {name}[{len(flat)}] __attribute__((aligned(8))) = {{\n{body}\n}};\n"


def rq_param(m0, sh, zp):
    """Parameter word of the rq / rq.a instructions (DESIGN.md D5)."""
    return (int(m0) & 0xFFFFFFFF) | ((int(sh) & 0x3F) << 32) | ((int(zp) & 0xFF) << 40)


def shape_str(t):
    return "x".join(str(int(d)) for d in t["shape"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--input", help="preprocessed NCHW float32 image (.npy) from prepare_data.py, or "
                    "'synthetic' (default: ~/aca/data/deit/eval/0000.npy if present, else synthetic)")
    ap.add_argument("--out", default="build/model")
    ap.add_argument("--golden", default="build/golden")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no report (export_model.py summarizes)")
    ap.add_argument("--range", help="FIRST:LAST op indices instead of a block (the stem, the tail)")
    ap.add_argument("--prefix", help="name prefix with --range, e.g. STEM; files are <prefix>_*.h")
    ap.add_argument("--base", type=int, default=10, help="with --range: op number used in names for FIRST")
    args = ap.parse_args(argv)

    buf = open(args.model, "rb").read()
    it = Interpreter(model_path=args.model,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                     experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    T = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()

    # Input image: explicit .npy, else the first eval image from get_model.sh, else synthetic.
    src = args.input                          # "synthetic" forces the synthetic image
    default = os.path.expanduser("~/aca/data/deit/eval/0000.npy")
    if src is None and os.path.exists(default):
        src = default
    if src and src != "synthetic":
        x = np.load(os.path.expanduser(src)).astype(np.float32)
    else:
        src = "synthetic (seed 0, smoothed N(0,1))"
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        k = np.ones(8, np.float32) / 8
        x = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 3, x)
        x = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 2, x) * 3
    inp = it.get_input_details()[0]
    s, zp = inp["quantization"]
    it.set_tensor(inp["index"], np.clip(np.round(x / s) + zp, -128, 127).astype(np.int8))
    it.invoke()

    b = args.block
    if args.range:
        start, end = (int(v) for v in args.range.split(":"))
        P, fname, base = args.prefix, args.prefix.lower(), args.base
    else:
        start, end = block_range(ops, b)
        P, fname, base = f"B{b}", f"block{b}", 10
    produced = {int(o) for op in ops for o in op["outputs"]}
    graph_inputs = {int(d["index"]) for d in it.get_input_details()}
    os.makedirs(args.out, exist_ok=True)
    gdir = os.path.join(args.golden, fname)
    os.makedirs(gdir, exist_ok=True)

    fc_roles = iter(["HEAD"] if args.range else ["QKV", "PROJ", "FC1", "FC2"])
    bmm_roles = iter(["QK", "AV"])
    sha = hashlib.sha256(buf).hexdigest()[:12]
    h = [f"// Generated by scripts/export_block.py from {os.path.basename(args.model)} (sha256 {sha}).",
         f"// DeiT-Tiny {'ops' if args.range else f'encoder block {b}: TFLite ops'} {start}..{end}. Do not edit by hand.",
         "// Multipliers: real = M0 * 2^(SHIFT-31), SHIFT > 0 = left shift (TFLite convention).",
         "// FC biases already include -zp_in * sum_k W[n][k] (zero-point folding, DESIGN.md D2).",
         f"// Op-level names: op k is OP{{k - {start - base}}}" + (" (block-0 numbering, so the same C code serves every block)." if not args.range else "."),
         f"#ifndef {P}_PARAMS_H", f"#define {P}_PARAMS_H", "#include <stdint.h>", ""]
    manifest = {"model": os.path.basename(args.model), "sha256": sha, "block": None if args.range else b,
                "ops": [start, end], "input_image": src, "tensors": {}, "op_list": []}
    saved = set()
    checks = []
    ew_checks = []                            # (op, name, integer model output, LiteRT output)
    op_golden = []                            # (role, [(suffix, tensor index)])

    def save(i):
        if i in saved or i < 0:
            return
        t = T[i]
        try:
            v = it.get_tensor(i)
        except ValueError:
            return
        name = f"t{i:04d}.npy"
        np.save(os.path.join(gdir, name), v)
        sc, zps = qparams(t)
        manifest["tensors"][str(i)] = {"name": t["name"], "file": name,
                                       "shape": [int(d) for d in t["shape"]],
                                       "dtype": np.dtype(t["dtype"]).name,
                                       "scale": sc if len(sc) <= 1 else f"{len(sc)} per-channel",
                                       "zero_point": zps[:1]}
        saved.add(i)

    k = start
    while k <= end:
        op = ops[k]
        name = op["op_name"]
        ins = [int(i) for i in op["inputs"] if i >= 0]
        outs = [int(i) for i in op["outputs"]]
        for i in ins + outs:
            save(i)
        manifest["op_list"].append({"index": k, "op": name, "inputs": ins, "outputs": outs})
        t_in = [T[i] for i in ins]
        t_out = T[outs[0]]
        so, zo = qparams(t_out)
        h.append(f"// ---- op {k} {name} (named OP{k - start + base}): in {' '.join(shape_str(t) for t in t_in)} -> out {shape_str(t_out)}")

        if name == "FULLY_CONNECTED":
            role = next(fc_roles)
            pre = f"{P}_{role}"
            op_golden.append((role, [("IN", ins[0]), ("OUT", outs[0])]))
            si, zi = qparams(t_in[0])
            sw, _ = qparams(t_in[1])
            W = it.get_tensor(ins[1]).astype(np.int64)          # [N][K]
            bias = it.get_tensor(ins[2]).astype(np.int64)
            n, kk = W.shape
            folded = bias - zi[0] * W.sum(axis=1)
            assert folded.min() >= INT32_MIN and folded.max() <= INT32_MAX
            ms = [quantize_multiplier(si[0] * sw[c] / so[0]) for c in range(n)]
            m0 = np.array([m for m, _ in ms], np.int64)
            sh = np.array([e for _, e in ms], np.int64)
            h += [f"#define {pre}_OP     {k}", f"#define {pre}_M      {int(np.prod(t_in[0]['shape'][:-1]))}",
                  f"#define {pre}_K      {kk}", f"#define {pre}_N      {n}",
                  f"#define {pre}_IN_ZP  ({zi[0]})", f"#define {pre}_OUT_ZP ({zo[0]})",
                  f"// W is row-major [N][K] (one output channel per row, K contiguous).",
                  c_array("int8_t", f"{pre}_W", W),
                  c_array("int32_t", f"{pre}_BIAS", folded),
                  c_array("int32_t", f"{pre}_M0", m0),
                  c_array("int8_t", f"{pre}_SHIFT", sh),
                  "// rq/rq.a parameter word per channel: M0 | shift << 32 | out_zp << 40 (D5)",
                  c_array("uint64_t", f"{pre}_RQP", [rq_param(m, e, zo[0]) for m, e in zip(m0, sh)])]
            if not args.no_check:
                a = it.get_tensor(ins[0]).reshape(-1, kk).astype(np.int64)
                acc = a @ W.T + folded
                checks.append((k, f"FC {role}", acc, m0, sh, zo[0],
                               it.get_tensor(outs[0]).reshape(acc.shape)))

        elif name == "BATCH_MATMUL":
            role = next(bmm_roles)
            pre = f"{P}_{role}"
            op_golden.append((role, [("LHS", ins[0]), ("RHS", ins[1]), ("OUT", outs[0])]))
            adj_x, adj_y = bmm_adj(buf, k)
            (s0, z0), (s1, z1) = qparams(t_in[0]), qparams(t_in[1])
            m0, sh = quantize_multiplier(s0[0] * s1[0] / so[0])
            h += [f"#define {pre}_OP     {k}", f"#define {pre}_ADJ_X  {int(adj_x)}",
                  f"#define {pre}_ADJ_Y  {int(adj_y)}  // 1: rhs is [.., N, K] (rows of rhs are dotted with rows of lhs)",
                  f"#define {pre}_LHS_ZP ({z0[0]})", f"#define {pre}_RHS_ZP ({z1[0]})",
                  f"#define {pre}_OUT_ZP ({zo[0]})", f"#define {pre}_M0     {m0}",
                  f"#define {pre}_SHIFT  ({sh})",
                  f"#define {pre}_RQP    0x{rq_param(m0, sh, zo[0]):012x}ull  // rq.a parameter word", ""]
            if not args.no_check:
                lhs = it.get_tensor(ins[0]).astype(np.int64) - z0[0]
                rhs = it.get_tensor(ins[1]).astype(np.int64) - z1[0]
                if adj_y:
                    rhs = np.swapaxes(rhs, -1, -2)
                acc = lhs @ rhs
                checks.append((k, f"BMM {role}", acc, m0, sh, zo[0], it.get_tensor(outs[0])))

        elif name == "DEQUANTIZE" and k + 1 <= end and ops[k + 1]["op_name"] == "SQUEEZE":
            # DEQUANTIZE -> SQUEEZE -> (later) QUANTIZE is an int8 -> int8 requantization.
            sq_out = ops[k + 1]["outputs"][0]
            qop = next(o for o in ops[k + 2:end + 1]
                       if o["op_name"] == "QUANTIZE" and o["inputs"][0] == sq_out)
            si, zi = qparams(t_in[0])
            sq, zq = qparams(T[qop["outputs"][0]])
            m0, sh = quantize_multiplier(si[0] / sq[0])
            pre = f"{P}_OP{k - start + base}_REQUANT"
            ident = int(si[0] == sq[0] and zi[0] == zq[0])
            h += [f"// fused with op {k + 1} SQUEEZE and op {qop['index']} QUANTIZE (float path in TFLite)",
                  f"#define {pre}_IDENTITY {ident}", f"#define {pre}_IN_ZP  ({zi[0]})",
                  f"#define {pre}_OUT_ZP ({zq[0]})", f"#define {pre}_M0     {m0}",
                  f"#define {pre}_SHIFT  ({sh})",
                  f"#define {pre}_RQP    0x{rq_param(m0, sh, zq[0]):012x}ull  // rq.a parameter word", ""]
            if not args.no_check:
                q = it.get_tensor(ins[0]).astype(np.int64) - zi[0]
                ref = it.get_tensor(qop["outputs"][0]).reshape(q.shape)
                checks.append((k, f"REQUANT op{k}->op{qop['index']}", q, m0, sh, zq[0], ref))

        elif name == "CONV_2D":
            # Patch embedding: a VALID convolution with stride = kernel size is an FC over the
            # non-overlapping patches. K is reordered to (c, kh, kw) so that im2col copies runs of
            # KW contiguous bytes from the NCHW input image.
            pre = f"{P}_CONV"
            op_golden.append(("CONV", [("OUT", outs[0])]))
            o = tflite.Conv2DOptions()
            t_opt = builtin_options(buf, k)[1]
            o.Init(t_opt.Bytes, t_opt.Pos)
            W4 = it.get_tensor(ins[1]).astype(np.int64)         # [N][KH][KW][Cin]
            n, kh, kw, cin = W4.shape
            assert (o.StrideH(), o.StrideW()) == (kh, kw) and o.Padding() == tflite.Padding.VALID
            assert o.FusedActivationFunction() == 0 and o.DilationHFactor() == o.DilationWFactor() == 1
            si, zi = qparams(t_in[0])
            sw, _ = qparams(t_in[1])
            _, hin, win, _ = (int(d) for d in t_in[0]["shape"])
            W = W4.transpose(0, 3, 1, 2).reshape(n, -1)
            bias = it.get_tensor(ins[2]).astype(np.int64)
            folded = bias - zi[0] * W.sum(axis=1)
            ms = [quantize_multiplier(si[0] * sw[c] / so[0]) for c in range(n)]
            m0 = np.array([m for m, _ in ms], np.int64)
            sh = np.array([e for _, e in ms], np.int64)
            h += [f"#define {pre}_OP     {k}", f"#define {pre}_IN_C   {cin}",
                  f"#define {pre}_IN_H   {hin}", f"#define {pre}_IN_W   {win}",
                  f"#define {pre}_KH     {kh}", f"#define {pre}_KW     {kw}",
                  f"#define {pre}_M      {(hin // kh) * (win // kw)}  // patches, row-major over the output grid",
                  f"#define {pre}_K      {W.shape[1]}", f"#define {pre}_N      {n}",
                  f"#define {pre}_IN_ZP  ({zi[0]})", f"#define {pre}_OUT_ZP ({zo[0]})",
                  "// W is [N][K] with K in (c, kh, kw) order (TFLite stores (kh, kw, c))",
                  c_array("int8_t", f"{pre}_W", W),
                  c_array("int32_t", f"{pre}_BIAS", folded),
                  c_array("int32_t", f"{pre}_M0", m0),
                  c_array("int8_t", f"{pre}_SHIFT", sh),
                  c_array("uint64_t", f"{pre}_RQP", [rq_param(m, e, zo[0]) for m, e in zip(m0, sh)])]
            if not args.no_check:
                x = it.get_tensor(ins[0]).astype(np.int64)[0]  # [H][W][C]
                A = x.reshape(hin // kh, kh, win // kw, kw, cin).transpose(0, 2, 4, 1, 3).reshape(-1, W.shape[1])
                acc = A @ W.T + folded
                checks.append((k, "CONV as FC", acc, m0, sh, zo[0],
                               it.get_tensor(outs[0]).reshape(acc.shape)))

        elif name == "QUANTIZE" and t_in[0]["dtype"] == np.int8:
            # int8 -> int8 QUANTIZE is a requantization
            si, zi = qparams(t_in[0])
            m0, sh = quantize_multiplier(si[0] / so[0])
            pre = f"{P}_OP{k - start + base}_REQUANT"
            h += [f"#define {pre}_IN_ZP  ({zi[0]})", f"#define {pre}_OUT_ZP ({zo[0]})",
                  f"#define {pre}_M0     {m0}", f"#define {pre}_SHIFT  ({sh})",
                  f"#define {pre}_RQP    0x{rq_param(m0, sh, zo[0]):012x}ull  // rq.a parameter word", ""]
            if not args.no_check:
                q = it.get_tensor(ins[0]).astype(np.int64) - zi[0]
                checks.append((k, f"REQUANT op{k}", q, m0, sh, zo[0], it.get_tensor(outs[0])))

        elif name == "BROADCAST_TO":
            assert ins[0] not in produced and list(t_in[0]["shape"]) == list(t_out["shape"])
            pre = f"{P}_OP{k - start + base}_BROADCAST"
            h += ["// a constant (the class token) broadcast to its own shape: no data movement",
                  c_array("int8_t", f"{pre}_CONST", it.get_tensor(ins[0])), ""]

        elif name == "CONCATENATION":
            qs = {(tuple(qparams(t)[0]), tuple(qparams(t)[1])) for t in t_in + [t_out]}
            assert len(qs) == 1, "CONCATENATION with requantization"
            h += [f"// CONCATENATION of {len(ins)} inputs, same quantization: data movement only", ""]

        elif name == "STRIDED_SLICE":
            consts = [it.get_tensor(i).tolist() for i in ins[1:]]
            h += [f"// STRIDED_SLICE begin/end/strides {consts}: data movement only", ""]

        elif name in ("ADD", "SUB", "MUL", "MEAN", "RSQRT", "SOFTMAX", "GELU"):
            # Integer-only formulations, checked against LiteRT below: ADD/SUB/MUL/MEAN as
            # fixed-point rescaling (ADD/SUB with single rounding, MUL/MEAN with gemmlowp double
            # rounding, as LiteRT's x86 kernels do), RSQRT/GELU as 256-entry tables, SOFTMAX with
            # an integer exp table and one division per row, plus an exact path for near-ties
            # (LiteRT's x86 SOFTMAX computes in float32).
            pre = f"{P}_OP{k - start + base}_{name}"
            for j, t in enumerate(t_in):
                if t["dtype"] != np.int8:
                    continue
                sc, zz = qparams(t)
                h.append(f"#define {pre}_IN{j}_ZP ({zz[0]})  // scale {sc[0]:.9g}")
                if ins[j] not in produced and ins[j] not in graph_inputs:     # constant tensor
                    h.append(c_array("int8_t", f"{pre}_IN{j}_CONST", it.get_tensor(ins[j])))
            h.append(f"#define {pre}_OUT_ZP ({zo[0]})  // scale {so[0]:.9g}")
            if name in ("ADD", "SUB"):
                s_a, s_b = qparams(t_in[0])[0][0], qparams(t_in[1])[0][0]
                twice = 2 * max(s_a, s_b)
                for j, sj in enumerate((s_a, s_b)):
                    m0, sh = quantize_multiplier(sj / twice)
                    h.append(f"#define {pre}_IN{j}_M0 {m0}\n#define {pre}_IN{j}_SHIFT ({sh})")
                m0, sh = quantize_multiplier(twice / ((1 << 20) * so[0]))
                h.append(f"#define {pre}_LEFT_SHIFT 20  // TFLite int8 ADD/SUB scheme, every rescaling single-rounded")
                h.append(f"#define {pre}_OUT_M0 {m0}\n#define {pre}_OUT_SHIFT ({sh})")
            elif name == "MUL":
                s_a, s_b = qparams(t_in[0])[0][0], qparams(t_in[1])[0][0]
                m0, sh = quantize_multiplier(s_a * s_b / so[0])
                h.append(f"#define {pre}_M0 {m0}\n#define {pre}_SHIFT ({sh})")
            elif name == "MEAN":
                n_red = int(t_in[0]["shape"][-1])
                m0, sh = mean_multiplier(qparams(t_in[0])[0][0], so[0], n_red)
                h.append(f"#define {pre}_N {n_red}  // reduced over the last axis")
                h.append(f"#define {pre}_M0 {m0}  // ~ s_in / (s_out * N), LiteRT's fixed-point division; applied to sum(x) - N*zp_in")
                h.append(f"#define {pre}_SHIFT ({sh})")
            elif name == "RSQRT":
                si, zi = qparams(t_in[0])
                h.append("// int8 -> int8 table indexed by (uint8_t)x, from LiteRT's integer RSQRT")
                h.append(c_array("int8_t", f"{pre}_LUT", rsqrt_lut(si[0], zi[0], so[0], zo[0])))
            elif name == "GELU":
                si, zi = qparams(t_in[0])
                o = tflite.GeluOptions()
                t_opt = builtin_options(buf, k)[1]
                o.Init(t_opt.Bytes, t_opt.Pos)
                h.append(f"// int8 -> int8 table indexed by (uint8_t)x (approximate={bool(o.Approximate())})")
                h.append(c_array("int8_t", f"{pre}_LUT", gelu_lut(si[0], zi[0], so[0], zo[0], o.Approximate())))
            elif name == "SOFTMAX":
                si, _ = qparams(t_in[0])
                o = tflite.SoftmaxOptions()
                t_opt = builtin_options(buf, k)[1]
                o.Init(t_opt.Bytes, t_opt.Pos)
                h.append(f"// exp(-s_in*beta*(255 - i)) * 2^{SOFTMAX_EXP_BITS}, beta = {o.Beta():g}; index 255 - max + x")
                h.append(c_array("uint32_t", f"{pre}_EXP", softmax_exp_table(si[0], o.Beta())))
                h.append("// the same table as IEEE float32 bit patterns, and the output scale as float32 bits")
                h.append(c_array("uint32_t", f"{pre}_EXPF", softmax_f32_table(si[0], o.Beta()).view(np.uint32)))
                h.append(f"#define {pre}_OUT_SCALE_F32 0x{int(np.float32(so[0]).view(np.uint32)):08x}u  // {so[0]:.9g}")
            h.append("")
            if not args.no_check:
                ref = it.get_tensor(outs[0]).astype(np.int64)
                xs = [it.get_tensor(i).astype(np.int64) for i in ins]
                (s0, z0) = [v[0] for v in qparams(t_in[0])]
                if name in ("ADD", "SUB", "MUL"):
                    (s1, z1) = [v[0] for v in qparams(t_in[1])]
                if name in ("ADD", "SUB"):
                    got = addsub_int8(xs[0], z0, s0, xs[1], z1, s1, so[0], zo[0], 1 if name == "ADD" else -1)
                elif name == "MUL":
                    got = rq_away((xs[0] - z0) * (xs[1] - z1), *quantize_multiplier(s0 * s1 / so[0]), zo[0])
                elif name == "MEAN":
                    n_red = xs[0].shape[-1]
                    got = rq_away(xs[0].sum(-1) - n_red * z0, *mean_multiplier(s0, so[0], n_red), zo[0])
                elif name == "RSQRT":
                    got = rsqrt_lut(s0, z0, so[0], zo[0])[xs[0] & 0xFF]
                elif name == "GELU":
                    got = gelu_lut(s0, z0, so[0], zo[0], o.Approximate())[xs[0] & 0xFF]
                else:
                    got, slow = softmax_int(xs[0], s0, o.Beta(), so[0], zo[0])
                    name = f"SOFTMAX ({slow} exact rows)"
                ew_checks.append((k, name, np.broadcast_to(got.reshape(ref.shape) if got.size == ref.size else got, ref.shape), ref))

        elif name in ("RESHAPE", "TRANSPOSE", "SLICE", "SQUEEZE", "QUANTIZE"):   # QUANTIZE: float in
            if name in ("TRANSPOSE", "SLICE"):
                consts = [it.get_tensor(i).tolist() for i in ins[1:]]
                h.append(f"// {name} {' '.join(str(c) for c in consts)}: data movement only")
            h.append("")
        else:
            raise SystemExit(f"op {k} {name} not handled")
        k += 1

    h += [f"#endif  // {P}_PARAMS_H", ""]
    with open(os.path.join(args.out, f"{fname}_params.h"), "w") as fh:
        fh.write("\n".join(h))

    blk_in = int(ops[start]["inputs"][0])
    blk_out = int(ops[end]["outputs"][0])
    gi, go = T[blk_in], T[blk_out]
    with open(os.path.join(args.out, f"{fname}_golden.h"), "w") as fh:
        fh.write(f"// Generated by scripts/export_block.py; input image: {src}\n")
        fh.write(f"#ifndef {P}_GOLDEN_H\n#define {P}_GOLDEN_H\n#include <stdint.h>\n")
        fh.write(f"#define {P}_INPUT \"{os.path.basename(src) if src.endswith('.npy') else src}\"\n")
        fh.write(f"#define {P}_IN_ZP ({qparams(gi)[1][0]})  // scale {qparams(gi)[0][0]:.9g}, shape {shape_str(gi)}\n")
        fh.write(f"#define {P}_OUT_ZP ({qparams(go)[1][0]})  // scale {qparams(go)[0][0]:.9g}, shape {shape_str(go)}\n")
        fh.write(c_array("int8_t", f"{P}_GOLDEN_IN", it.get_tensor(blk_in)))
        fh.write(c_array("int8_t", f"{P}_GOLDEN_OUT", it.get_tensor(blk_out)))
        fh.write(f"#endif  // {P}_GOLDEN_H\n")
    for role, tens in op_golden:
        g = f"{P}_GOLDEN_{role}"
        with open(os.path.join(args.out, f"{fname}_golden_{role.lower()}.h"), "w") as fh:
            fh.write(f"// Generated by scripts/export_block.py; input image: {src}\n")
            fh.write(f"#ifndef {g}_H\n#define {g}_H\n#include <stdint.h>\n")
            for suffix, i in tens:
                fh.write(f"// {suffix}: tensor {i} '{T[i]['name']}', shape {shape_str(T[i])}\n")
                fh.write(c_array("int8_t", f"{g}_{suffix}", it.get_tensor(i)))
            fh.write(f"#endif  // {g}_H\n")
    manifest["block_input"], manifest["block_output"] = blk_in, blk_out
    with open(os.path.join(gdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    if not args.quiet:
        print(f"{P if args.range else f'block {b}'}: ops {start}..{end}, input image: {src}")
        print(f"wrote {args.out}/{fname}_params.h, {args.out}/{fname}_golden*.h, {gdir}/ ({len(saved)} tensors)")
    # % bit-exact of the formulation the C code uses: FC-like ops round ties toward +inf (rq),
    # BATCH_MATMUL and requantizations away from zero (rq.a); elementwise ops as listed
    summary = []
    for k, label, acc, m0, sh, zo, ref in checks:
        f = mbqm_halfup if label.startswith(("FC", "CONV")) else mbqm
        summary.append((k, label, 100 * np.mean(np.clip(f(acc, m0, sh) + zo, -128, 127) == ref.astype(np.int64))))
    for k, label, got, ref in ew_checks:
        summary.append((k, label, 100 * np.mean(got == ref)))
    if args.quiet:
        return summary
    if checks:
        print("\ninteger reference vs TFLite golden, % of outputs bit-exact (max |diff|)")
        print(f"  {'':<30}{'ties away from 0':>18}{'ties toward +inf':>18}")
        for k, label, acc, m0, sh, zo, ref in checks:
            cols = []
            for f in (mbqm, mbqm_halfup):
                d = np.clip(f(acc, m0, sh) + zo, -128, 127) - ref.astype(np.int64)
                cols.append(f"{100 * np.mean(d == 0):9.3f}% ({int(np.abs(d).max())})")
            print(f"  op {k:3d} {label:<23}{cols[0]:>18}{cols[1]:>18}")
    if ew_checks:
        print("\nelementwise / table ops (integer formulation used in C) vs TFLite golden")
        for k, label, got, ref in ew_checks:
            d = got - ref
            print(f"  op {k:3d} {label:<23}{100 * np.mean(d == 0):9.3f}% ({int(np.abs(d).max())})")
    return summary


if __name__ == "__main__":
    main()
