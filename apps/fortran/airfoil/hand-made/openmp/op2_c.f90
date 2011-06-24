

module OP2_C

	use, intrinsic :: ISO_C_BINDING

	integer, parameter :: MAX_NAME_LEN = 100

	! accessing operation codes
	integer(c_int) :: OP_READ =		1
	integer(c_int) :: OP_WRITE =	2
	integer(c_int) :: OP_INC =		3
	integer(c_int) :: OP_RW =			4

	type, BIND(C) :: op_set

		integer(kind=c_int) :: size		! number of elements in the set
		integer(kind=c_int) :: index	! index into list of sets (OP_list_set)
		type(c_ptr) ::				 name	  ! set name
		
	end type op_set

	type, BIND(C) :: op_map

		type(op_set) ::														from	! set map from
		type(op_set) ::														to		! set map to
		integer(kind=c_int) ::										dim		! dimension of map
		integer(kind=c_int) ::										index	! index into list of maps (OP_list_map)
		type(c_ptr) ::														map		! array defining map
		type(c_ptr) ::														name	! mapping name
		
	end type op_map

	type, BIND(C) :: op_dat

		type(op_set) ::														set		! set on which data is defined
		integer(kind=c_int) ::										dim		! dimension of data	
		integer(kind=c_int) ::										index	! index into list of datasets (OP_list_dat)
		integer(kind=c_int) ::										size	! size of each element in dataset
		type(c_ptr) ::														dat		! data on host
		type(c_ptr) ::														dat_d ! data on device (in the CUDA implementation this changes)
		type(c_ptr) ::														type	! data type
		type(c_ptr) ::														name	! name of dataset

	end type op_dat


	type, BIND(C) :: op_plan
		
		! input arguments
		type(c_ptr) ::														name	
		integer(kind=c_int) ::										set_index, nargs
		type(c_ptr) ::														arg_idxs, idxs, map_idxs, dims
		type(c_ptr) ::														typs
		type(c_ptr) ::														accs
		
		! execution plan
		type(c_ptr) ::														nthrcol ! number of thread colors for each block
		type(c_ptr) ::													  thrcol ! thread colors
		type(c_ptr) ::														offset ! offset for primary set
		type(c_ptr) ::														ind_maps ! pointers for indirect datasets
		type(c_ptr) ::														nindirect ! size of ind_maps (for Fortran)
		type(c_ptr) ::														ind_offs ! offsets for indirect datasets
		type(c_ptr) ::														ind_sizes ! offsets for indirect datasets
		type(c_ptr) ::														maps ! regular pointers, renumbered as needed
		type(c_ptr) ::														nelems ! number of elements in each block
		integer(kind=c_int) ::										ncolors ! number of block colors
		type(c_ptr) ::														ncolblk  ! number of blocks for each color
		integer(kind=c_int)	::										nblocks ! number of blocks (for Fortran)
		type(c_ptr) ::														blkmap ! block mapping
		integer(kind=c_int) ::										nshared	! bytes of shared memory required
		real(kind=c_float) ::											transfer ! bytes of data transfer per kernel call
		real(kind=c_float) ::											transfer2 ! bytes of cache line per kernel call
		
	end type op_plan


	! declaration of identity map
	type(op_map) :: OP_ID, OP_GBL

	! Declarations of op_par_loop implemented in C
	interface

			subroutine op_decl_set_f ( setsize, set, name ) BIND(C,name='op_decl_set')

					use, intrinsic :: ISO_C_BINDING
					
					import :: op_set
					
					integer(kind=c_int), value, intent(in) :: setsize
					type(op_set) :: set
					character(kind=c_char,len=1) ::	name	
					
			end subroutine op_decl_set_f

			subroutine op_decl_map_f ( from, to, mapdim, data, map, name ) BIND(C,name='op_decl_map_f')

					use, intrinsic :: ISO_C_BINDING

					import :: op_set, op_map

					type(op_set), intent(in) :: from, to					
					integer(kind=c_int), value, intent(in) :: mapdim
					type(c_ptr), intent(in) :: data
					type(op_map), intent(out) :: map
					character(kind=c_char,len=1) ::	name	
					
			end subroutine op_decl_map_f

			subroutine op_decl_null_map ( map ) BIND(C,name='op_decl_null_map')

					use, intrinsic :: ISO_C_BINDING

					import :: op_map

					type(op_map), intent(out) :: map
					
			end subroutine op_decl_null_map

			subroutine op_decl_dat_f ( set, datdim, type, datsize, dat, data, name ) BIND(C,name='op_decl_dat_f')

					use, intrinsic :: ISO_C_BINDING

					import :: op_set, op_dat

					type(op_set) ::									set				
					integer(kind=c_int), value ::		datdim, datsize
					character(kind=c_char,len=1) ::	type
					type(c_ptr) ::									dat
					type(op_dat) ::									data
					character(kind=c_char,len=1) ::	name
												
			end subroutine op_decl_dat_f
	
			subroutine op_decl_const_f ( constdim, dat, name ) BIND(C,name='op_decl_const_f')
		
				use, intrinsic :: ISO_C_BINDING

				integer(kind=c_int), value :: constdim
				type(c_ptr), intent(in) :: dat
				character(kind=c_char,len=1) ::	name
		
			end subroutine op_decl_const_f
	
      subroutine op_timers ( elapsedTime ) BIND(C,name='op_timers')

        use, intrinsic :: ISO_C_BINDING
        
        real(kind=c_double) :: elapsedTime
      
      end subroutine op_timers
  
		! debug C functions (to obtain similar output file that can be diff-ed
		subroutine op_par_loop_2_f ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1  &
													 & ) BIND(C,name='op_par_loop_2')

			use, intrinsic :: ISO_C_BINDING
			
			import :: op_set, op_map, op_dat

!			external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface
		
			type(op_set) :: set
			type(op_dat) :: data0, data1
			integer(kind=c_int), value :: itemSel0, itemSel1, access0, access1
			type(op_map) :: map0, map1
			
		end subroutine op_par_loop_2_f

		subroutine op_par_loop_5_F ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4 &
													 & ) BIND(C,name='op_par_loop_5')

			use, intrinsic :: ISO_C_BINDING
			
			import :: op_set, op_map, op_dat

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface
			
!			type(c_funptr) :: subroutineName
			type(op_set) :: set
			type(op_dat) :: data0, data1, data2, data3, data4
			integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4
			integer(kind=c_int), value :: access0, access1, access2, access3, access4
			type(op_map) :: map0, map1, map2, map3, map4
			
		end subroutine op_par_loop_5_F


		subroutine op_par_loop_6_F ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4, &
														 & data5, itemSel5, map5, access5  &
													 & ) BIND(C,name='op_par_loop_6')

			use, intrinsic :: ISO_C_BINDING
			
			import :: op_set, op_map, op_dat

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface
			
!			type(c_funptr) :: subroutineName
			type(op_set) :: set
			type(op_dat) :: data0, data1, data2, data3, data4, data5
			integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5 
			integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5
			type(op_map) :: map0, map1, map2, map3, map4, map5
			
		end subroutine op_par_loop_6_F

		subroutine op_par_loop_8_F ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4, &
														 & data5, itemSel5, map5, access5, &
														 & data6, itemSel6, map6, access6, &
														 & data7, itemSel7, map7, access7  &
													 & ) BIND(C,name='op_par_loop_8')

			use, intrinsic :: ISO_C_BINDING
			
			import :: op_set, op_map, op_dat

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface
			
!			type(c_funptr) :: subroutineName
			type(op_set) :: set
			type(op_dat) :: data0, data1, data2, data3, data4, data5, data6, data7
			integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5, itemSel6, itemSel7
			integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5, access6, access7
			type(op_map) :: map0, map1, map2, map3, map4, map5, map6, map7
			
		end subroutine op_par_loop_8_F

		type(c_ptr) function cplan ( name, setId, argsNumber, args, idxs, maps, accs, indsNumber, inds ) BIND(C)

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

		end function cplan


	
		! debug C functions (to obtain similar output file that can be diff-ed
		integer(KIND=C_INT) function openfile ( filename ) BIND(C)

			use, intrinsic :: ISO_C_BINDING
			character(c_char), dimension(20) :: filename
		
		end function openfile

		integer(KIND=C_INT) function closefile ( ) BIND(C)
		
				use, intrinsic :: ISO_C_BINDING
		
		end function closefile

		
		integer(KIND=C_INT) function writerealtofile ( dataw ) BIND(C)
		
			use, intrinsic :: ISO_C_BINDING

			real(c_double) :: dataw

			
		end function writerealtofile

		integer(KIND=C_INT) function writeinttofile ( dataw, datasize, filename ) BIND(C)
		
			use, intrinsic :: ISO_C_BINDING

			type(c_ptr) :: dataw 
			integer(c_int) :: datasize
			character(c_char), dimension(20) :: filename
			
		end function writeinttofile

		end interface

		interface op_decl_dat
			module procedure op_decl_dat_real_8, op_decl_dat_integer_4
		end interface op_decl_dat
			
		interface op_decl_gbl
			module procedure op_decl_gbl_real_8 !, op_decl_gbl_integer_4 ! not needed for now
		end	interface op_decl_gbl		

		interface op_decl_const
			module procedure op_decl_const_integer_4, op_decl_const_real_8, op_decl_const_scalar_integer_4, &
										 & op_decl_const_scalar_real_8 
		end	interface op_decl_const


	contains

	subroutine op_init ()
		
		call op_decl_null_map ( OP_ID )
		call op_decl_null_map ( OP_GBL )
		
		OP_ID%dim = 0 ! OP_ID code used in arg_set
		OP_GBL%dim = -1 ! OP_GBL code used in arg_set
		
	end subroutine op_init

	subroutine op_decl_set ( setsize, set, opname )

		integer(kind=c_int), value, intent(in) :: setsize
		type(op_set) :: set
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname
		
		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		
		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_set_F ( setsize, set, fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_set_F ( setsize, set, cname )
		end if
		
	end	subroutine op_decl_set

	subroutine op_decl_map ( from , to, mapdim, dat, outmap, opname )

		type(op_set), intent(in) :: from, to
		integer, intent(in) :: mapdim
		integer(4), dimension(*), intent(in), target :: dat
		type(op_map), intent(out) :: outmap
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_map_F ( from, to, mapdim, c_loc ( dat )	, outmap, fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_map_F ( from, to, mapdim, c_loc ( dat )	, outmap, cname )
		end if
			
	end	subroutine op_decl_map

	subroutine op_decl_dat_real_8 ( set, datdim, dat, data, opname )

		type(op_set), intent(in) :: set
		integer, intent(in) :: datdim
		real(8), dimension(*), intent(in), target :: dat
		type(op_dat), intent(out) :: data			
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname
		
		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR


		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_dat_f ( set, datdim, type, 8, c_loc ( dat ), data, fakeName )
		else
			cname = opname//char(0)
			call op_decl_dat_f ( set, datdim, type, 8, c_loc ( dat ), data, cname )
		end if
					
	end	subroutine op_decl_dat_real_8
	
	subroutine op_decl_dat_integer_4(set, datdim, dat, data, opname)
		type(op_set), intent(in) :: set
		integer, intent(in) :: datdim
		integer(4), dimension(*), intent(in), target :: dat
		type(op_dat), intent(out) :: data
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_dat_f ( set, datdim, type, 4, c_loc ( dat ), data, fakeName )
		else
			cname = opname//char(0)
			call op_decl_dat_f ( set, datdim, type, 4, c_loc ( dat ), data, cname )
		end if

	end	subroutine op_decl_dat_integer_4
	
	
	subroutine op_decl_gbl_real_8 ( dat, gbldim, data)
		
		real(8), dimension(*), intent(in), target :: dat
		integer, intent(in) :: gbldim
		type(op_dat), intent(out) :: data

		! unused name
		character(kind=c_char,len=7) ::	name = C_CHAR_'NONAME'//C_NULL_CHAR
			character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

		! unsed op_set
		type(op_set) unusedSet;
		
		call op_decl_dat_f ( unusedSet, gbldim, type, 8, c_loc ( dat ), data, name )
		
	end subroutine op_decl_gbl_real_8
	
	subroutine op_decl_const_integer_4 ( constdim, dat, opname )
				
		integer(kind=c_int), value :: constdim
		integer(4), dimension(*), intent(in), target :: dat
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_const_F ( constdim, c_loc ( dat ), cname )
	  end if

	end	subroutine op_decl_const_integer_4

	subroutine op_decl_const_real_8 ( constdim, dat, opname )
				
		integer(kind=c_int), value :: constdim
		real(8), dimension(*), intent(in), target :: dat
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_const_F ( constdim, c_loc ( dat ), cname )
	  end if

	end	subroutine op_decl_const_real_8


	subroutine op_decl_const_scalar_integer_4 ( constdim, dat, opname )
				
		integer(kind=c_int), value :: constdim
		integer(4), intent(in), target :: dat
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_const_F ( constdim, c_loc ( dat ), cname )
	  end if

	end	subroutine op_decl_const_scalar_integer_4
	
	subroutine op_decl_const_scalar_real_8 ( constdim, dat, opname )
				
		integer(kind=c_int), value :: constdim
		real(8), intent(in), target :: dat
		character(len=*), optional :: opname
		character(len=len(opname)+1) :: cname

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
		else
			cname = C_CHAR_''//opname//C_NULL_CHAR
			call op_decl_const_F ( constdim, c_loc ( dat ), cname )
	  end if

	end	subroutine op_decl_const_scalar_real_8
	
	
	
	
	
	
	subroutine op_par_loop_2 ( subroutineName, set, &
													 & data0, itemSel0, map0, access0, &
													 & data1, itemSel1, map1, access1 &
												 & )
														
    external subroutineName
													
		type(op_set) :: set
		type(op_dat) :: data0, data1
		integer(kind=c_int) :: itemSel0, itemSel1
		integer(kind=c_int) :: access0, access1
		type(op_map) :: map0, map1

		integer(kind=c_int) :: itemSelC0, itemSelC1

		! selector are used in C++ to address correct map field, hence must be converted from 1->N style to 0->N-1 one
		itemSelC0 = itemSel0 - 1
		itemSelC1 = itemSel1 - 1

		! warning: look at the -1 on itemSels: it is used to access C++ arrays!
		call op_par_loop_2_f ( subroutineName, set, &
											 & data0, itemSelC0, map0, access0, &
											 & data1, itemSelC1, map1, access1 &
										 & )
										 
	end subroutine op_par_loop_2

	subroutine op_par_loop_5 ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4  &
													 & )												 			
		
		external subroutineName

		type(op_set) :: set
		type(op_dat) :: data0, data1, data2, data3, data4
		integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4
		integer(kind=c_int) :: access0, access1, access2, access3, access4
		type(op_map) :: map0, map1, map2, map3, map4

		integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4

		! see above
		itemSelC0 = itemSel0 - 1
		itemSelC1 = itemSel1 - 1
		itemSelC2 = itemSel2 - 1
		itemSelC3 = itemSel3 - 1
		itemSelC4 = itemSel4 - 1

		! warning: look at the -1 on itemSels: it is used to access C++ arrays!
		call op_par_loop_5_f ( subroutineName, set, &
											 & data0, itemSelC0, map0, access0, &
											 & data1, itemSelC1, map1, access1, &
											 & data2, itemSelC2, map2, access2, &
											 & data3, itemSelC3, map3, access3, &
											 & data4, itemSelC4, map4, access4  &
										 & )
										 
	end subroutine op_par_loop_5

	
	subroutine op_par_loop_6 ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4, &
														 & data5, itemSel5, map5, access5  &
													 & )

		external subroutineName
															
		type(op_set) :: set
		type(op_dat) :: data0, data1, data2, data3, data4, data5
		integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5 
		integer(kind=c_int) :: access0, access1, access2, access3, access4, access5
		type(op_map) :: map0, map1, map2, map3, map4, map5

		integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4, itemSelC5 

		itemSelC0 = itemSel0 - 1
		itemSelC1 = itemSel1 - 1
		itemSelC2 = itemSel2 - 1
		itemSelC3 = itemSel3 - 1
		itemSelC4 = itemSel4 - 1
		itemSelC5 = itemSel5 - 1 

		! warning: look at the -1 on itemSels: it is used to access C++ arrays!
		call op_par_loop_6_f ( subroutineName, set, &
											 & data0, itemSelC0, map0, access0, &
											 & data1, itemSelC1, map1, access1, &
											 & data2, itemSelC2, map2, access2, &
											 & data3, itemSelC3, map3, access3, &
											 & data4, itemSelC4, map4, access4, &
											 & data5, itemSelC5, map5, access5  &
										 & )
										 
	end subroutine op_par_loop_6

	
	subroutine op_par_loop_8 ( subroutineName, set, &
														 & data0, itemSel0, map0, access0, &
														 & data1, itemSel1, map1, access1, &
														 & data2, itemSel2, map2, access2, &
														 & data3, itemSel3, map3, access3, &
														 & data4, itemSel4, map4, access4, &
														 & data5, itemSel5, map5, access5, &
														 & data6, itemSel6, map6, access6, &
														 & data7, itemSel7, map7, access7  &
													 & )

    external subroutineName																	
!			type(c_funptr) :: subroutineName
		type(op_set) :: set
		type(op_dat) :: data0, data1, data2, data3, data4, data5, data6, data7
		integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5, itemSel6, itemSel7
		integer(kind=c_int) :: access0, access1, access2, access3, access4, access5, access6, access7
		type(op_map) :: map0, map1, map2, map3, map4, map5, map6, map7

		integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4, itemSelC5, itemSelC6, itemSelC7

		itemSelC0 = itemSel0 - 1
		itemSelC1 = itemSel1 - 1
		itemSelC2 = itemSel2 - 1
		itemSelC3 = itemSel3 - 1
		itemSelC4 = itemSel4 - 1
		itemSelC5 = itemSel5 - 1
		itemSelC6 = itemSel6 - 1 
		itemSelC7 = itemSel7 - 1 

		! warning: look at the -1 on itemSels: it is used to access C++ arrays!
		call op_par_loop_8_f ( subroutineName, set, &
											 & data0, itemSelC0, map0, access0, &
											 & data1, itemSelC1, map1, access1, &
											 & data2, itemSelC2, map2, access2, &
											 & data3, itemSelC3, map3, access3, &
											 & data4, itemSelC4, map4, access4, &
											 & data5, itemSelC5, map5, access5, &
											 & data6, itemSelC6, map6, access6, &
											 & data7, itemSelC7, map7, access7  &
										 & )
											 
		end subroutine op_par_loop_8
	
end module OP2_C

