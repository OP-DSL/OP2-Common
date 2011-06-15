! This module defines the interoperable data types between OP2 C and Fortran
! and it defines the Fortran interface for declaration routines


module OP2_Fortran_Declarations

  use, intrinsic :: ISO_C_BINDING

  integer, parameter :: MAX_NAME_LEN = 100
  integer, parameter :: BSIZE_DEFAULT = 256

	! accessing operation codes
	integer(c_int) :: OP_READ = 1
	integer(c_int) :: OP_WRITE = 2
	integer(c_int) :: OP_INC = 3
	integer(c_int) :: OP_RW = 4

	type, BIND(C) :: op_set_core

		integer(kind=c_int) :: index	! position in the private OP2 array of op_set_core variables
		integer(kind=c_int) :: size	! number of elements in the set
		type(c_ptr)         :: name	! set name
		
	end type op_set_core

	type op_set

		type (op_set_core), pointer :: setPtr
		type(c_ptr)                 :: setCptr

	end type op_set

	type, BIND(C) :: op_map_core

		integer(kind=c_int) :: 		index	! position in the private OP2 array of op_map_core variables
		type(c_ptr) ::						from	! set map from
		type(c_ptr) ::						to		! set map to
		integer(kind=c_int) ::		dim		! dimension of map
		type(c_ptr) ::						map		! array defining map
		type(c_ptr) ::						name	! map name				
	
	end type op_map_core

	type op_map

	type(op_map_core), pointer :: mapPtr
	type(c_ptr) :: mapCptr

	end type op_map	

	type, BIND(C) :: op_dat_core

		integer(kind=c_int) :: 		index	! position in the private OP2 array of op_dat_core variables
		type(c_ptr) ::						set		! set on which data is defined
		integer(kind=c_int) ::		dim		! dimension of data	
		integer(kind=c_int) ::		size	! size of each element in dataset
		type(c_ptr) ::						dat		! data on host
		type(c_ptr) ::	  				dat_d ! data on device
		type(c_ptr) ::						type	! data type
		type(c_ptr) ::						name	! data name

	end type op_dat_core

	type op_dat

		type(op_dat_core), pointer :: dataPtr
		type(c_ptr) :: dataCptr
		
	end type op_dat

!  type, BIND(C) :: op_arg
  
!    integer(4) ::   index
!    type(op_dat) :: dat
!    type(op_map) :: map
!    integer(4) ::   dim
!    integer(4) ::   idx
!    integer(4) ::   size
!    type(c_ptr) ::  data
!    type(c_ptr) ::  data_d
!    type(c_ptr) ::  type
!    integer(4) ::   acc
!    integer(4) ::   argType
  
!  end type op_arg

	! declaration of identity and global mapping
	type(op_map) :: OP_ID, OP_GBL

	! Declarations of op_par_loop implemented in C
	interface

			subroutine op_init_core ( argc, argv, diags ) BIND(C,name='op_init_core')

				use, intrinsic :: ISO_C_BINDING

				integer(kind=c_int), intent(in), value :: argc
				type(c_ptr), intent(in)								 :: argv
				integer(kind=c_int), intent(in), value :: diags

			end subroutine op_init_core

			type(c_ptr) function op_decl_set_f ( setsize, name ) BIND(C,name='op_decl_set_f')

					use, intrinsic :: ISO_C_BINDING
					
					import :: op_set_core
					
					integer(kind=c_int), value, intent(in)		:: setsize					
					character(kind=c_char,len=1), intent(in)	:: name				
          
			end function op_decl_set_f

			type(c_ptr) function op_decl_map_f ( from, to, mapdim, data, name ) BIND(C,name='op_decl_map_f')

					use, intrinsic :: ISO_C_BINDING

					type(c_ptr), value, intent(in) :: from, to					
					integer(kind=c_int), value, intent(in) :: mapdim
					type(c_ptr), intent(in) :: data
					character(kind=c_char,len=1), intent(in) ::	name	
          
			end function op_decl_map_f

			type(c_ptr) function op_decl_null_map () BIND(C,name='op_decl_null_map')

					use, intrinsic :: ISO_C_BINDING
					
			end function op_decl_null_map

			type(c_ptr) function op_decl_dat_f ( set, datdim, type, datsize, dat, name ) BIND(C,name='op_decl_dat_f')

					use, intrinsic :: ISO_C_BINDING

					import :: op_set_core, op_dat_core

					type(c_ptr), value, intent(in)					 :: set				
					integer(kind=c_int), value							 ::	datdim, datsize
					character(kind=c_char,len=1), intent(in) ::	type
					type(c_ptr), intent(in)									 :: dat
					character(kind=c_char,len=1), intent(in) ::	name
          
			end function op_decl_dat_f
	
			subroutine op_decl_const_f ( constdim, dat, name ) BIND(C,name='op_decl_const_f')
		
				use, intrinsic :: ISO_C_BINDING

				integer(kind=c_int), value :: constdim
				type(c_ptr), intent(in) :: dat
				character(kind=c_char,len=1) ::	name
		
			end subroutine op_decl_const_f

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

	subroutine op_init ( diags )

    ! formal parameter
		integer(4) :: diags
		
    ! local variables
		integer(4) :: argc = 0

		type (op_map_core), pointer :: idPtr
		type (op_map_core), pointer :: gblPtr

    ! calling C function
		OP_ID%mapCPtr = op_decl_null_map ()
		OP_GBL%mapCPtr = op_decl_null_map ()

    ! idptr and gblPtr are required because of a gfortran compiler internal error
    call c_f_pointer ( OP_ID%mapCPtr, idPtr )
    call c_f_pointer ( OP_GBL%mapCPtr, gblPtr )

		idPtr%dim = 0 ! OP_ID code used in arg_set
		gblPtr%dim = -1 ! OP_GBL code used in arg_set

		OP_ID%mapPtr => idPtr
		OP_GBL%mapPtr => gblPtr

		call op_init_core ( argc, C_NULL_PTR, diags )
		
	end subroutine op_init

	subroutine op_decl_set ( setsize, set, opname )

		integer(kind=c_int), value, intent(in) :: setsize
		type(op_set) :: set
		character(len=*), optional :: opName
		    
    character(kind=c_char,len = len ( opname ) + 1) :: cName
		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		
		type(c_ptr) :: setCPtr = C_NULL_PTR
		type(op_set_core), pointer :: setFPtr
  
      	
		if ( present ( opname ) .eqv. .false. ) then
			set%setCPtr = op_decl_set_F ( setsize, fakeName )
		else
#ifdef GNU_FORTRAN
			cName = C_CHAR_''//opName//C_NULL_CHAR
#else
			cname = opname//char(0)
#endif

			set%setCPtr = op_decl_set_F ( setsize, cName )
		end if
		
		! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
		call c_f_pointer ( set%setCPtr, set%setPtr )

	end	subroutine op_decl_set

	subroutine op_decl_map ( from, to, mapdim, dat, map, opname )

		type(op_set), intent(in) :: from, to
		integer, intent(in) :: mapdim
		integer(4), dimension(*), intent(in), target :: dat
		type(op_map) :: map
		character(len=*), optional :: opName

    character(kind=c_char,len = len ( opname ) + 1) :: cName
		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

		if ( present ( opname ) .eqv. .false. ) then
			map%mapCPtr = op_decl_map_F ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), fakeName )
		else
#ifdef GNU_FORTRAN
			cName = C_CHAR_''//opName//C_NULL_CHAR
#else
			cname = opname//char(0)
#endif
			map%mapCPtr = op_decl_map_F ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), cName )
		end if
			
		! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
		call c_f_pointer ( map%mapCPtr, map%mapPtr )
		
	end	subroutine op_decl_map

	subroutine op_decl_dat_real_8 ( set, datdim, dat, data, opname )

		type(op_set), intent(in) :: set
		integer, intent(in) :: datdim
		real(8), dimension(*), intent(in), target :: dat
		type(op_dat) :: data
		character(len=*), optional :: opName
		
    character(kind=c_char,len = len ( opname ) + 1) :: cName

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

		type(c_ptr) :: dataCPtr = C_NULL_PTR

		if ( present ( opname ) .eqv. .false. ) then
			data%dataCPtr = op_decl_dat_f ( set%setCPtr, datdim, type, 8, c_loc ( dat ), fakeName )
		else
#ifdef GNU_FORTRAN
			cName = C_CHAR_''//opName//C_NULL_CHAR
#else
			cname = opname//char(0)
#endif

			data%dataCPtr = op_decl_dat_f ( set%setCPtr, datdim, type, 8, c_loc ( dat ), cName )
		end if
			
		! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
		call c_f_pointer ( data%dataCPtr, data%dataPtr )			

	end	subroutine op_decl_dat_real_8
	
	subroutine op_decl_dat_integer_4 ( set, datdim, dat, data, opname )
		type(op_set), intent(in) :: set
		integer, intent(in) :: datdim
		integer(4), dimension(*), intent(in), target :: dat
		type(op_dat) :: data
		character(len=*), optional :: opname
		
		character(kind=c_char,len = len ( opname ) + 1) :: cName

		character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
		character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

		type(c_ptr) :: dataCPtr = C_NULL_PTR

		if ( present ( opname ) .eqv. .false. ) then
			data%dataCPtr = op_decl_dat_f ( set%setCPtr, datdim, type, 4, c_loc ( dat ), fakeName )
		else
#ifdef GNU_FORTRAN
			cName = C_CHAR_''//opName//C_NULL_CHAR
#else
			cname = opname//char(0)
#endif
			data%dataCPtr = op_decl_dat_f ( set%setCPtr, datdim, type, 4, c_loc ( dat ), cName )
		end if

		! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
		call c_f_pointer ( data%dataCPtr, data%dataPtr )

	end	subroutine op_decl_dat_integer_4
	
	
	subroutine op_decl_gbl_real_8 ( dat, gbldata, gbldim )
		
		real(8), dimension(*), intent(in), target :: dat
		type(op_dat) :: gblData
		integer, intent(in) :: gbldim

		! unused name
		character(kind=c_char,len=7) ::	name = C_CHAR_'NONAME'//C_NULL_CHAR
		character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

		! unsed op_set
		type(op_set_core), target :: unusedSet;
		
		type(c_ptr) :: gblCPtr = C_NULL_PTR
		
		gblData%dataCPtr = op_decl_dat_f ( c_loc ( unusedSet ), gbldim, type, 8, c_loc ( dat ), name )
		
		call c_f_pointer ( gblData%dataCPtr, gblData%dataPtr )
		
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
			cname = opname//char(0)
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
			cname = opname//char(0)
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
			cname = opname//char(0)
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
			cname = opname//char(0)
			call op_decl_const_F ( constdim, c_loc ( dat ), cname )
	  end if

	end	subroutine op_decl_const_scalar_real_8
	
end module OP2_Fortran_Declarations

