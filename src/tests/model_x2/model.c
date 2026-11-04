// Whole DeiT-Tiny W8A8 inference on the X2 outer-product prototype (DESIGN.md D10): src/tests/model
// with X2 defined (qop4, or qop8 with -DQOP8).
//   make -B TEST=model_x2 run EXT=xqmac                       4x4 accumulators
//   make -B TEST=model_x2 run EXT=xqmac OPT="-O2 -DQOP8"      8x8 accumulators
#define X2
#include "../model/model.c"
