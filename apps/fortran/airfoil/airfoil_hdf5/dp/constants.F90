MODULE OP2_CONSTANTS

#ifdef OP2_WITH_CUDAFOR
  use cudafor
  real(8), constant :: gam_OP2
  real(8), constant :: gm1_OP2
  real(8), constant :: cfl_OP2
  real(8), constant :: eps_OP2
  real(8), constant :: mach_OP2
  real(8), constant :: alpha_OP2
  real(8), constant :: air_const_OP2
  real(8), constant :: qinf_OP2(4)
  real(8) :: gam, gm1, cfl, eps, mach, alpha, qinf(4)
#else

#ifdef OP2_WITH_OMP4
!$omp declare target(gam, gm1, cfl, eps, mach, alpha, qinf)
#endif

real(8) :: gam, gm1, cfl, eps, mach, alpha, qinf(4)
!$acc declare create(gam, gm1, cfl, eps, mach, alpha, qinf(4))

#endif

END MODULE OP2_CONSTANTS
