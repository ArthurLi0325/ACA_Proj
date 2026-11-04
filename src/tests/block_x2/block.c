// Whole block 0 on the X2 outer-product prototype (DESIGN.md D10): src/tests/block with X2
// defined, so FC and BATCH_MATMUL use qop4 (default) or qop8 (-DQOP8) and every producer writes
// the K-interleaved operand layout (src/common/vit.h).
//   make -B TEST=block_x2 run EXT=xqmac                       4x4 accumulators
//   make -B TEST=block_x2 run EXT=xqmac OPT="-O2 -DQOP8"      8x8 accumulators
#define X2
#include "../block/block.c"
