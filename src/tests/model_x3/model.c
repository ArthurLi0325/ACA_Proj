// Whole DeiT-Tiny W8A8 inference on X3, the row-vector dot products (DESIGN.md D12): src/tests/model
// with X2 and X3 defined.
//   make -B TEST=model_x3 run EXT=xqmac VEC=1                     vqdot4.vx
//   make -B TEST=model_x3 run EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"   vqdot8.vx
#define X2
#define X3
#include "../model/model.c"
