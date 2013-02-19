MODULE OP2_CONSTANTS

#ifdef OP2_WITH_CUDAFOR
  use cudafor
  real(8), constant :: gam
  real(8), constant :: gm1
  real(8), constant :: cfl
  real(8), constant :: eps
  real(8), constant :: mach
  real(8), constant :: alpha
  real(8), constant :: air_const
  real(8), constant :: qinf(4)
#else
real(8) :: gam, gm1, cfl, eps, mach, alpha, qinf(4)
#endif

END MODULE OP2_CONSTANTS
