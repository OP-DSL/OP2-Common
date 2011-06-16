! This module defines the interoperable data structures of the OP2 RT support
! (i.e. plan struct) and defines the interface for the C plan function
  
module OP2_Fortran_RT_Support

  use, intrinsic :: ISO_C_BINDING

  integer(kind=c_int), parameter :: F_OP_ARG_DAT = 0
  integer(kind=c_int), parameter :: F_OP_ARG_GBL = 1

 	type, BIND(C) :: op_plan

		! input arguments
		type(c_ptr) ::														name	
		type(c_ptr) ::                            set
    integer(kind=c_int) ::                    nargs, ninds, part_size
		type(c_ptr) ::														maps
		type(c_ptr) ::														idxs
		type(c_ptr) ::														accs

		! execution plan
		type(c_devptr) ::													nthrcol ! number of thread colors for each block
		type(c_devptr) ::													thrcol ! thread colors
		type(c_devptr) ::													offset ! offset for primary set
		type(c_ptr) ::														ind_maps ! pointers for indirect datasets
		type(c_devptr) ::													ind_offs ! offsets for indirect datasets
		type(c_devptr) ::													ind_sizes ! offsets for indirect datasets
		type(c_ptr) ::														nindirect ! size of ind_maps (for Fortran)
		type(c_ptr) ::														maps ! regular pointers, renumbered as needed
		integer(kind=c_int)	::										nblocks ! number of blocks (for Fortran)
		type(c_devptr) ::													nelems ! number of elements in each block
		integer(kind=c_int) ::										ncolors ! number of block colors
		type(c_ptr) ::														ncolblk  ! number of blocks for each color
		type(c_devptr) ::													blkmap ! block mapping
		integer(kind=c_int) ::										nshared	! bytes of shared memory required
		real(kind=c_float) ::											transfer ! bytes of data transfer per kernel call
		real(kind=c_float) ::											transfer2 ! bytes of cache line per kernel call
    integer(kind=c_int) ::                    count ! number fo times called (should not work for fortran?)
	end type op_plan

	interface

		! C wrapper to plan function for Fortran (cPlan function)
		type(c_ptr) function cplan ( name, setId, argsNumber, args, idxs, maps, accs, indsNumber, inds, argsType, partitionSize ) &
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

      integer(kind=c_int) :: argsType(*)
      integer(kind=c_int) :: partitionSize

		end function cplan

	end interface

end module OP2_Fortran_RT_Support
