! This module defines the interoperable data structures of the OP2 RT support
! (i.e. plan struct) and defines the interface for the C plan function

module OP2_Fortran_RT_Support

  use, intrinsic :: ISO_C_BINDING

  integer(kind=c_int), parameter :: F_OP_ARG_DAT = 0
  integer(kind=c_int), parameter :: F_OP_ARG_GBL = 1

  type, BIND(C) :: op_plan

    ! input arguments
    type(c_ptr) ::         name
    type(c_ptr) ::         set
    integer(kind=c_int) :: nargs, ninds, part_size
    type(c_ptr) ::         in_maps
    type(c_ptr) ::         dats
    type(c_ptr) ::         idxs
    type(c_ptr) ::         accs

    ! execution plan
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      nthrcol ! number of thread colors for each block
    type(c_devptr) ::      thrcol  ! thread colors
    type(c_devptr) ::      offset  ! offset for primary set
#else
    type(c_ptr) ::         nthrcol ! number of thread colors for each block
    type(c_ptr) ::         thrcol  ! thread colors
    type(c_ptr) ::         offset  ! offset for primary set
#endif

#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      ind_map ! concatenated pointers for indirect datasets
#else
    type(c_ptr) ::         ind_map ! concatenated pointers for indirect datasets
#endif

    type(c_ptr) ::         ind_maps ! pointers for indirect datasets
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      ind_offs ! offsets for indirect datasets
    type(c_devptr) ::      ind_sizes ! offsets for indirect datasets
#else
    type(c_ptr) ::         ind_offs ! offsets for indirect datasets
    type(c_ptr) ::         ind_sizes ! offsets for indirect datasets
#endif
    type(c_ptr) ::         nindirect ! size of ind_maps (for Fortran)

#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      loc_map ! concatenated maps to local indices, renumbered as needed
#else
    type(c_ptr) ::         loc_map ! concatenated maps to local indices, renumbered as needed
#endif

    type(c_ptr) ::         maps ! maps to local indices, renumbered as needed
    integer(kind=c_int) :: nblocks ! number of blocks (for Fortran)
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      nelems ! number of elements in each block
#else
    type(c_ptr) ::         nelems ! number of elements in each block
#endif
    integer(kind=c_int) :: ncolors_core ! mumber of core colors in MPI
    integer(kind=c_int) :: ncolors_owned ! number of colors in MPI for blocks that only have owned elements
    integer(kind=c_int) :: ncolors ! number of block colors
    type(c_ptr) ::         ncolblk  ! number of blocks for each color

#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      blkmap ! block mapping
#else
    type(c_ptr) ::         blkmap ! block mapping
#endif

    type(c_ptr) ::         nsharedCol ! bytes of shared memory required per block colour
    integer(kind=c_int) :: nshared ! bytes of shared memory required
    real(kind=c_float) ::  transfer ! bytes of data transfer per kernel call
    real(kind=c_float) ::  transfer2 ! bytes of cache line per kernel call
    integer(kind=c_int) :: count ! number fo times called (should not work for fortran?)

  end type op_plan

  interface

    ! C wrapper to plan function for Fortran (cPlan function)
    type(c_ptr) function FortranPlanCallerCUDA ( name, &
                                               & setId, &
                                               & argsNumber, &
                                               & args, &
                                               & idxs, &
                                               & maps, &
                                               & accs, &
                                               & indsNumber, &
                                               & inds, &
                                               & argsType, &
                                               & partitionSize ) &
              & BIND(C,name='FortranPlanCallerCUDA')

      use, intrinsic :: ISO_C_BINDING

      character(kind=c_char) :: name(*) ! name of kernel
      integer(kind=c_int), value :: setId ! position in OP_set_list of the related set
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      integer(kind=c_int) :: args(*) ! positions in OP_dat_list of arguments to op_par_loop
      integer(kind=c_int) :: idxs(*) ! array of indexes to maps
      integer(kind=c_int) :: maps(*) ! positions in OP_map_list of arguments to op_par_loop
      integer(kind=c_int) :: accs(*) ! access flags to arguments
      integer(kind=c_int), value :: indsNumber ! number of arguments accessed indirectly via a map

      ! indexes for indirectly accessed arguments (same indrectly accessed argument = same index)
      integer(kind=c_int), dimension(*) :: inds

      integer(kind=c_int) :: argsType(*)
      integer(kind=c_int), value :: partitionSize

    end function FortranPlanCallerCUDA

    ! C wrapper to plan function for Fortran (cPlan function)
    type(c_ptr) function cplan_OpenMP ( name, &
                                      & setId, &
                                      & argsNumber, &
                                      & args, &
                                      & idxs, &
                                      & maps, &
                                      & accs, &
                                      & indsNumber, &
                                      & inds, &
                                      & argsType, &
                                      & partitionSize &
                                      )  BIND(C,name='FortranPlanCallerOpenMP')

      use, intrinsic :: ISO_C_BINDING

      character(kind=c_char) :: name(*) ! name of kernel
      integer(kind=c_int), value :: setId ! position in OP_set_list of the related set
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      integer(kind=c_int) :: args(*) ! positions in OP_dat_list of arguments to op_par_loop
      integer(kind=c_int) :: idxs(*) ! array of indexes to maps
      integer(kind=c_int) :: maps(*) ! positions in OP_map_list of arguments to op_par_loop
      integer(kind=c_int) :: accs(*) ! access flags to arguments
      integer(kind=c_int), value :: indsNumber ! number of arguments accessed indirectly via a map

      ! indexes for indirectly accessed arguments (same indrectly accessed argument = same index)
      integer(kind=c_int), dimension(*) :: inds

      integer(kind=c_int) :: argsType(*)
      integer(kind=c_int), value :: partitionSize

    end function cplan_OpenMP

  end interface

end module OP2_Fortran_RT_Support

