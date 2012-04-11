! This file defines the module used by all OP2 back-ends (e.g. CUDA and openmp)
! and it makes use of the proper implementation in C
! (e.g. op_cuda_decl.c or op_openmp_decl.cpp)
!
! It defines the interoperable data types between OP2 C and Fortran
! and it defines the Fortran interface for declaration routines

module OP2_Fortran_Declarations

  use, intrinsic :: ISO_C_BINDING
#ifdef OP2_WITH_CUDAFOR
  use cudafor
#endif

  integer, parameter :: MAX_NAME_LEN = 100
  integer, parameter :: BSIZE_DEFAULT = 256

  ! accessing operation codes
  integer(c_int) :: OP_READ = 1
  integer(c_int) :: OP_WRITE = 2
  integer(c_int) :: OP_INC = 3
  integer(c_int) :: OP_RW = 4
  integer(c_int) :: OP_MIN = 4
  integer(c_int) :: OP_MAX = 5

  type, BIND(C) :: op_set_core

    integer(kind=c_int) :: index        ! position in the private OP2 array of op_set_core variables
    integer(kind=c_int) :: size         ! number of elements in the set
    type(c_ptr)         :: name         ! set name
    integer(kind=c_int) :: exec_size    ! number of additional imported elements to be executed
    integer(kind=c_int) :: nonexec_size ! number of additional imported elements that are not executed

  end type op_set_core

  type :: op_set

    type (op_set_core), pointer :: setPtr
    type(c_ptr)                 :: setCptr

  end type op_set

  type, BIND(C) :: op_map_core

    integer(kind=c_int) ::    index ! position in the private OP2 array of op_map_core variables
    type(c_ptr) ::            from  ! set map from
    type(c_ptr) ::            to    ! set map to
    integer(kind=c_int) ::    dim   ! dimension of map
    type(c_ptr) ::            map   ! array defining map
    type(c_ptr) ::            name  ! map name

  end type op_map_core

  type :: op_map

    type(op_map_core), pointer :: mapPtr
    type(c_ptr) :: mapCptr

  end type op_map

  type, BIND(C) :: op_dat_core

    integer(kind=c_int) ::    index    ! position in the private OP2 array of op_dat_core variables
    type(c_ptr) ::            set      ! set on which data is defined
    integer(kind=c_int) ::    dim      ! dimension of data
    integer(kind=c_int) ::    size     ! size of each element in dataset
    type(c_ptr) ::            dat      ! data on host
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::         dat_d    ! data on device
#else
    type(c_ptr) ::            dat_d    ! data on device
#endif
    type(c_ptr) ::            type     ! data type
    type(c_ptr) ::            name     ! data name
    type(c_ptr) ::            buffer_d ! buffer for MPI halo sends on the device

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
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags

    end subroutine op_init_core

    type(c_ptr) function op_decl_set_c ( setsize, name ) BIND(C,name='op_decl_set')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core

      integer(kind=c_int), value, intent(in)    :: setsize
      character(kind=c_char,len=1), intent(in)  :: name

    end function op_decl_set_c

    type(c_ptr) function op_decl_map_c ( from, to, mapdim, data, name ) BIND(C,name='op_decl_map')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: from, to
      integer(kind=c_int), value, intent(in)   :: mapdim
      type(c_ptr), intent(in), value           :: data
      character(kind=c_char,len=1), intent(in) :: name

    end function op_decl_map_c

    type(c_ptr) function op_decl_null_map () BIND(C,name='op_decl_null_map')

      use, intrinsic :: ISO_C_BINDING

    end function op_decl_null_map

    type(c_ptr) function op_decl_dat_c ( set, datdim, type, datsize, dat, name ) BIND(C,name='op_decl_dat')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_dat_core

      type(c_ptr), value, intent(in)           :: set
      integer(kind=c_int), value               :: datdim, datsize
      character(kind=c_char,len=1), intent(in) :: type
      type(c_ptr), intent(in), value           :: dat
      character(kind=c_char,len=1), intent(in) :: name

    end function op_decl_dat_c

    subroutine op_decl_const_c ( constdim, dat, name ) BIND(C,name='op_decl_const_char')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value     :: constdim
      type(c_ptr), intent(in), value :: dat
      character(kind=c_char,len=1)   :: name

    end subroutine op_decl_const_c

    type(c_ptr) function op_decl_gbl_f ( dataIn, dataDim, dataSize, name ) BIND(C,name='op_decl_gbl_f')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), intent(in) :: dataIn
      integer(kind=c_int), value, intent(in) :: dataDim, dataSize
      character(kind=c_char,len=1) :: name

    end function op_decl_gbl_f

    subroutine op_fetch_data_f ( opdat ) BIND(C,name='op_fetch_data')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine op_fetch_data_f

    subroutine op_timers_f ( cpu, et ) BIND(C,name='op_timers')
      use, intrinsic :: ISO_C_BINDING

      real(kind=c_double) :: cpu, et

    end subroutine op_timers_f

    function get_set_size ( set ) BIND(C,name='get_set_size')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core

      integer(kind=c_int) get_set_size

      type(op_set_core) :: set

    end function

    function get_associated_set_size_f ( dat ) BIND(C,name='get_associated_set_size')
      use, intrinsic :: ISO_C_BINDING

      import :: op_dat_core

      integer(kind=c_int) :: get_associated_set_size_f

      type(op_dat_core) :: dat

    end function

    subroutine op_get_dat ( opdat ) BIND(C,name='op_get_dat')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine

    subroutine op_put_dat ( opdat ) BIND(C,name='op_put_dat')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine

   subroutine dumpOpDatFromDevice_c ( data, label, sequenceNumber ) BIND(C,name='dumpOpDatFromDevice')
      use, intrinsic :: ISO_C_BINDING

     import :: op_dat_core

     type(op_dat_core) :: data
     character(len=1,kind=c_char) :: label
     integer(kind=c_int) :: sequenceNumber

   end subroutine

   subroutine dumpOpDat_c ( data, fileName ) BIND(C,name='dumpOpDat')
      use, intrinsic :: ISO_C_BINDING

     import :: op_dat_core

     type(op_dat_core) :: data
     character(len=1,kind=c_char) :: fileName

   end subroutine

   subroutine dumpOpMap_c ( map, fileName ) BIND(C,name='dumpOpMap')
      use, intrinsic :: ISO_C_BINDING

     import :: op_map_core

     type(op_map_core) :: map
     character(len=1,kind=c_char) :: fileName

   end subroutine

    subroutine op_diagnostic_output (  ) BIND(C,name='op_diagnostic_output')
      use, intrinsic :: ISO_C_BINDING
    end subroutine

  end interface

  ! the two numbers at the end of the name indicate the size of the type (e.g. real(8))
  interface op_decl_dat
    module procedure op_decl_dat_real_8, op_decl_dat_integer_4, &
                     op_decl_dat_real_8_2, op_decl_dat_integer_4_2, &
                     op_decl_dat_real_8_3, op_decl_dat_integer_4_3
  end interface op_decl_dat

  interface op_decl_gbl
    module procedure op_decl_gbl_real_8,  op_decl_gbl_integer_4_scalar
  end interface op_decl_gbl

  interface op_decl_const
    module procedure op_decl_const_integer_4, op_decl_const_real_8, op_decl_const_scalar_integer_4, &
    & op_decl_const_scalar_real_8
  end interface op_decl_const

contains

  subroutine op_init ( diags )

    ! formal parameter
    integer(4) :: diags

    ! local variables
    integer(4) :: argc = 0

#ifdef OP2_WITH_CUDAFOR
    integer(4) :: setDevReturnVal = -1
    integer(4) :: devPropRetVal = -1
    type(cudadeviceprop) :: deviceProperties
#endif

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

#ifdef OP2_WITH_CUDAFOR
    ! support for GTX
    setDevReturnVal = cudaSetDevice ( 3 )

    devPropRetVal = cudaGetDeviceProperties ( deviceProperties, 3 )

    print *, 'Using: ', deviceProperties%name
#endif
  end subroutine op_init

  subroutine op_decl_set ( setsize, set, opname )

    integer(kind=c_int), value, intent(in) :: setsize
    type(op_set) :: set
    character(kind=c_char,len=*), optional :: opName

    if ( present ( opname ) ) then
      set%setCPtr = op_decl_set_c ( setsize, opname//char(0) )
    else
      set%setCPtr = op_decl_set_c ( setsize, C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
    call c_f_pointer ( set%setCPtr, set%setPtr )

  end subroutine op_decl_set

  subroutine op_decl_map ( from, to, mapdim, dat, map, opname )

    type(op_set), intent(in) :: from, to
    integer, intent(in) :: mapdim
    integer(4), dimension(*), intent(in), target :: dat
    type(op_map) :: map
    character(kind=c_char,len=*), optional :: opName

    if ( present ( opname ) ) then
      map%mapCPtr = op_decl_map_c ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), opname//C_NULL_CHAR )
    else
      map%mapCPtr = op_decl_map_c ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( map%mapCPtr, map%mapPtr )

  end subroutine op_decl_map

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   declarations of op_dats    !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine op_decl_dat_real_8 ( set, datdim, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opName

    character(kind=c_char,len=7) :: type = C_CHAR_'double'//C_NULL_CHAR

    if ( present ( opname ) ) then
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type, 8, c_loc ( dat ), opName//C_NULL_CHAR )
    else
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type, 8, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( data%dataCPtr, data%dataPtr )

  end subroutine op_decl_dat_real_8

  subroutine op_decl_dat_real_8_2 ( set, datdim, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opName

    call op_decl_dat_real_8 ( set, datdim, dat, data, opname )

  end subroutine op_decl_dat_real_8_2

  subroutine op_decl_dat_real_8_3 ( set, datdim, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(:,:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opName

    call op_decl_dat_real_8 ( set, datdim, dat, data, opname )

  end subroutine op_decl_dat_real_8_3

  subroutine op_decl_dat_integer_4 ( set, datdim, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname

    character(kind=c_char,len=4) :: type = C_CHAR_'int'//C_NULL_CHAR

    if ( present ( opname ) ) then
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type, 4, c_loc ( dat ), opName//C_NULL_CHAR )
    else
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type, 4, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( data%dataCPtr, data%dataPtr )

  end subroutine op_decl_dat_integer_4

  subroutine op_decl_dat_integer_4_2 ( set, datdim, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname

    call op_decl_dat_integer_4 ( set, datdim, dat, data, opname )

  end subroutine op_decl_dat_integer_4_2

  subroutine op_decl_dat_integer_4_3 ( set, datdim, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(:,:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname

    call op_decl_dat_integer_4 ( set, datdim, dat, data, opname )

  end subroutine op_decl_dat_integer_4_3

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   declarations of globals    !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine op_decl_gbl_real_8 ( dat, gbldata, gbldim )

    real(8), dimension(*), intent(in), target :: dat
    type(op_dat) :: gblData
    integer, intent(in) :: gbldim

    ! FIXME: should this be double?
    character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

    gblData%dataCPtr = op_decl_gbl_f ( c_loc ( dat ), gbldim, 8, type )

    call c_f_pointer ( gblData%dataCPtr, gblData%dataPtr )

  end subroutine op_decl_gbl_real_8

  subroutine op_decl_gbl_integer_4_scalar ( dat, gbldata)

    integer(4), intent(in), target :: dat
    type(op_dat) :: gblData

    character(kind=c_char,len=8) :: type = C_CHAR_'integer'//C_NULL_CHAR

    gblData%dataCPtr = op_decl_gbl_f ( c_loc ( dat ), 1, 4, type )

    call c_f_pointer ( gblData%dataCPtr, gblData%dataPtr )

  end subroutine op_decl_gbl_integer_4_scalar

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   declarations of constants  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine op_decl_const_integer_4 ( dat, constdim, opname )

    integer(4), dimension(*), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    if ( present ( opname ) ) then
      call op_decl_const_c ( constdim, c_loc ( dat ), opname//char(0) )
    else
      call op_decl_const_c ( constdim, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

  end subroutine op_decl_const_integer_4

  subroutine op_decl_const_real_8 ( dat, constdim, opname )

    real(8), dimension(*), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    if ( present ( opname ) ) then
      call op_decl_const_c ( constdim, c_loc ( dat ), opname//char(0) )
    else
      call op_decl_const_c ( constdim, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

  end subroutine op_decl_const_real_8

  subroutine op_decl_const_scalar_integer_4 ( dat, constdim, opname )

    integer(4), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    if ( present ( opname ) ) then
      call op_decl_const_c ( constdim, c_loc ( dat ), opname//char(0) )
    else
      call op_decl_const_c ( constdim, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

  end subroutine op_decl_const_scalar_integer_4

  subroutine op_decl_const_scalar_real_8 ( dat, constdim, opname )

    real(8), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    if ( present ( opname ) ) then
      call op_decl_const_c ( constdim, c_loc ( dat ), opname//char(0) )
    else
      call op_decl_const_c ( constdim, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

  end subroutine op_decl_const_scalar_real_8

  subroutine op_fetch_data ( opdat )

    type(op_dat) :: opdat

    call op_fetch_data_f ( opdat%dataPtr)

  end subroutine op_fetch_data

  subroutine op_timers ( et )

    real(kind=c_double) :: et

    real(kind=c_double) :: cpu = 0

    call op_timers_f ( cpu, et )

  end subroutine op_timers

  function dumpOpDat ( dat, fileName )

    integer(4) :: dumpOpDat
    type(op_dat) :: dat
    character(len=*) :: fileName

    call dumpOpDat_c ( dat%dataPtr, fileName )

    dumpOpDat = 0

  end function dumpOpDat

  function dumpOpMap ( map, fileName )

    integer(4) :: dumpOpMap
    type(op_map) :: map
    character(len=*) :: fileName

    call dumpOpMap_c ( map%mapPtr, fileName )

    dumpOpMap = 0

  end function dumpOpMap

  function dumpOpDatFromDevice ( dat, label, sequenceNumber )

    integer(4) :: dumpOpDatFromDevice

    type(op_dat) :: dat
    character(len=*) :: label
    integer(4) :: sequenceNumber

    call dumpOpDatFromDevice_c ( dat%dataPtr, label, sequenceNumber )

    dumpOpDatFromDevice = 0

  end function dumpOpDatFromDevice

  function get_associated_set_size ( dat )

    integer(kind=c_int) :: get_associated_set_size

    type(op_dat) :: dat

    get_associated_set_size = get_associated_set_size_f ( dat%dataPtr )

  end function

end module OP2_Fortran_Declarations

