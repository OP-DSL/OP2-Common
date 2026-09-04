#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX
#endif

[[maybe_unused]] static inline OP_FUN_PREFIX double maxfun(double a, double b) {
   return a>b ? a : b;
}
