// The FC kernel this test links: B1 (scalar, software requantization) with -DSCALAR,
// otherwise X1 with qmac8 + rq (src/tests/fc_x1, USE_RQ).
#ifdef SCALAR
#include "../fc_b1/fc.c"
#else
#define USE_RQ
#include "../fc_x1/fc.c"
#endif
