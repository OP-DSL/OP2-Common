
! This file defines the configuration variables used by the Fortran CUDA OP2 back-end

module cudaConfigurationParams

  ! Default block size = number of threads in a single block
  integer(4) :: FOP_BLOCK_SIZE = 512

  ! fixed for architecture kind (compute capability?)
  integer(4) :: OP_WARP_SIZE = 32

end module cudaConfigurationParams
