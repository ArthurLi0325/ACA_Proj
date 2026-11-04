#!/usr/bin/env python3
"""Instruction mix and per-function breakdown from a Spike PC histogram.

Spike's `-g` flag prints "<pc-hex> <count>" for every executed PC. Joining that
with the objdump disassembly gives the dynamic count of every instruction
without writing a (huge) commit log.

usage: insn_mix.py <prog.dump> <prog.hist> [--macs N] [--csv out.csv]
  prog.dump : objdump -d -M no-aliases,numeric output (made by the Makefile)
"""
import argparse
import collections
import re
import sys

LOADS = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "lwsp", "ldsp"}
STORES = {"sb", "sh", "sw", "sd", "swsp", "sdsp"}
MULDIV = {"mul", "mulh", "mulhsu", "mulhu", "mulw",
          "div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}
JUMP = {"jal", "jalr", "j", "jr"}
CSR = {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci", "ecall", "ebreak", "mret", "wfi", "fence", "fence.i"}
XQMAC = {0: "qmac8", 1: "dot8", 4: "rq", 5: "rq.a"}  # custom-0 funct3 -> mnemonic (DESIGN.md D6)
XQOP = {0: "qop4", 1: "qop8", 2: "qacc.col", 3: "qacc.rq", 4: "qacc.rqa", 5: "qacc.row"}  # custom-1 (D10)
CATEGORIES = ["load", "store", "alu", "mul/div", "branch", "jump", "csr/sys", "xqmac", "custom/other"]

# Startup/console code; reported separately so it does not skew workload metrics.
RUNTIME_FUNCS = {"_start", "trap_entry", "trap_handler", "printf", "print_num",
                 "putchar_htif", "con_flush", "htif_syscall", "exit"}

FUNC_RE =re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)")


def classify(mnem):
    if mnem in XQMAC.values() or mnem in XQOP.values():
        return "xqmac"
    base = mnem[2:] if mnem.startswith("c.") else mnem
    if base in LOADS:
        return "load"
    if base in STORES:
        return "store"
    if base in MULDIV:
        return "mul/div"
    if base in BRANCH:
        return "branch"
    if base in JUMP:
        return "jump"
    if base in CSR:
        return "csr/sys"
    if base.startswith("f") or base.startswith(".insn") or base == "unknown":
        return "custom/other"
    return "alu"


def parse_dump(path):
    insns = {}  # pc -> (mnemonic, encoding_bytes, function)
    func = "?"
    for line in open(path):
        m = FUNC_RE.match(line)
        if m:
            # fold GCC clones (foo.constprop.0, foo.isra.0, foo.part.0, foo.cold) into foo
            func = re.sub(r"\.(constprop|isra|part|cold|lto_priv)(\.\d+)*$", "", m.group(2))
            func = re.sub(r"\.(constprop|isra|part|cold|lto_priv)(\.\d+)*", "", func)
            continue
        m = INSN_RE.match(line)
        if m:
            mnem, enc = m.group(3), int(m.group(2), 16)
            # objdump prints the custom instructions as .insn/unknown; name them from the encoding
            if (mnem.startswith(".insn") or mnem == "unknown") and enc & 0xFE00007F == 0x0B:
                mnem = XQMAC.get((enc >> 12) & 7, mnem)
            elif (mnem.startswith(".insn") or mnem == "unknown") and enc & 0x7F == 0x2B:
                mnem = XQOP.get((enc >> 12) & 7, mnem)
            insns[int(m.group(1), 16)] = (mnem, len(m.group(2)) // 2, func)
    return insns


def parse_hist(path):
    hist = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]+", parts[0]) and parts[1].isdigit():
            hist[int(parts[0], 16)] = int(parts[1])
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("hist")
    ap.add_argument("--macs", type=int, default=0, help="MAC count of the workload")
    ap.add_argument("--csv", help="write per-mnemonic counts to CSV")
    ap.add_argument("--funcs", help="comma-separated functions to include (default: all but runtime)")
    args = ap.parse_args()

    insns = parse_dump(args.dump)
    hist = parse_hist(args.hist)
    only = set(args.funcs.split(",")) if args.funcs else None

    cat = collections.Counter()
    mnem = collections.Counter()
    func = collections.Counter()
    compressed = 0
    unknown_pc = 0
    runtime = 0
    for pc, n in hist.items():
        if pc not in insns:          # e.g. Spike boot ROM at 0x1000
            unknown_pc += n
            continue
        m, nbytes, f = insns[pc]
        if (only is not None and f not in only) or (only is None and f in RUNTIME_FUNCS):
            runtime += n
            continue
        cat[classify(m)] += n
        mnem[m] += n
        func[f] += n
        if nbytes == 2:
            compressed += n

    total = sum(cat.values())
    if total == 0:
        sys.exit("no executed instructions matched the disassembly")

    print(f"Dynamic instructions : {total:,}  (excluded: {runtime:,} runtime/filtered, {unknown_pc} outside ELF)")
    print(f"Compressed (16-bit)  : {compressed:,} ({100 * compressed / total:.1f}%)")
    mem = cat["load"] + cat["store"]
    print(f"Memory instructions  : {mem:,} ({100 * mem / total:.1f}%)")
    if args.macs:
        print(f"MACs                 : {args.macs:,}")
        print(f"MAC / instruction    : {args.macs / total:.3f}")
        print(f"Load+store / MAC     : {mem / args.macs:.3f}")

    print("\nInstruction mix")
    for c in CATEGORIES:
        print(f"  {c:<13}{cat[c]:>14,}  {100 * cat[c] / total:6.2f}%")

    print("\nPer function (operator attribution)")
    for f, n in func.most_common(15):
        print(f"  {f:<30}{n:>14,}  {100 * n / total:6.2f}%")

    print("\nTop mnemonics")
    for m, n in mnem.most_common(12):
        print(f"  {m:<13}{n:>14,}  {100 * n / total:6.2f}%")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("mnemonic,category,count\n")
            for m, n in mnem.most_common():
                fh.write(f"{m},{classify(m)},{n}\n")


if __name__ == "__main__":
    main()
