

! Module declaring the plan data structure for CUDA.
! The difference with the plain data structure definition
! is in the device pointers used for some structures.
! This may change in the future to reflect a common
! pointer data structure which will relieve us
! from distinguishing

module Plan_CUDA

  use cudafor

  type, BIND(C) :: op_plan_cuda

    ! input arguments
    type(c_ptr) ::                            name
    type(c_ptr) ::                            set
    integer(kind=c_int) ::                    nargs, ninds, part_size
    type(c_ptr) ::                            in_maps
    type(c_ptr) ::                            dats

    type(c_ptr) ::                            idxs
    type(c_ptr) ::                            accs

    ! execution plan
    type(c_devptr) ::                         nthrcol ! number of thread colors for each block
    type(c_devptr) ::                         thrcol ! thread colors
    type(c_devptr) ::                         offset ! offset for primary set
    type(c_ptr) ::                            ind_maps ! pointers for indirect datasets
    type(c_devptr) ::                         ind_offs ! offsets for indirect datasets
    type(c_devptr) ::                         ind_sizes ! offsets for indirect datasets
    type(c_ptr) ::                            nindirect ! size of ind_maps (for Fortran)
    type(c_ptr) ::                            maps ! regular pointers, renumbered as needed
    integer(kind=c_int) ::                    nblocks ! number of blocks (for Fortran)
    type(c_devptr) ::                         nelems ! number of elements in each block
    integer(kind=c_int) ::                    ncolors ! number of block colors
    type(c_ptr) ::                            ncolblk  ! number of blocks for each color
    type(c_devptr) ::                         blkmap ! block mapping
    integer(kind=c_int) ::                    nshared ! bytes of shared memory required
    real(kind=c_float) ::                     transfer ! bytes of data transfer per kernel call
    real(kind=c_float) ::                     transfer2 ! bytes of cache line per kernel call
    integer(kind=c_int) ::                    count ! number fo times called (should not work for fortran?)

  end type op_plan_cuda


end module Plan_CUDA