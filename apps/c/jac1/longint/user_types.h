#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX
#endif

static inline OP_FUN_PREFIX double maxfun(double a, double b) {
   return a>b ? a : b;
}
