// Whole block 0 on X3, the row-vector dot products (DESIGN.md D12): src/tests/block with X2 and X3
// defined, so FC and BATCH_MATMUL use vqdot4.vx (default) or vqdot8.vx (-DQDOT8) and every producer
// writes the X3 operand layout (src/common/vit.h).
//   make -B TEST=block_x3 run EXT=xqmac VEC=1                     vqdot4.vx
//   make -B TEST=block_x3 run EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"   vqdot8.vx
#define X2
#define X3
#include "../block/block.c"
