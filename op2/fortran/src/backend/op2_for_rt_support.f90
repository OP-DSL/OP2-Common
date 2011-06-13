! This module defines the interoperable data structures of the OP2 RT support
! (i.e. plan struct) and defines the interface for the C plan function


module OP2_Fortran_RT_Support

  ! not needed for now
  ! use OP2_Fortran_Declarations

  interface

    ! C wrapper to plan function for Fortran (cPlan function)
    type(c_ptr) function cplan ( name, setId, argsNumber, args, idxs, maps, accs, indsNumber, inds ) &
              & BIND(C,name='FortranPlanCaller')

      use, intrinsic :: ISO_C_BINDING

      import :: c_int, c_ptr

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

    end function cplan

  end interface

end module OP2_Fortran_RT_Support
