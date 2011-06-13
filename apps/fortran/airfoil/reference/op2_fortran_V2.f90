module OP2_Fortran

  use, intrinsic :: ISO_C_BINDING

  integer, parameter :: MAX_NAME_LEN = 100
  integer, parameter :: BSIZE_DEFAULT = 256


  ! accessing operation codes
  integer(c_int) :: OP_READ =   1
  integer(c_int) :: OP_WRITE =  2
  integer(c_int) :: OP_INC =    3
  integer(c_int) :: OP_RW =     4

  type, BIND(C) :: op_set_core

    integer(kind=c_int) ::    size  ! number of elements in the set
    type(c_ptr) ::            name  ! set name

  end type op_set_core

  type op_set

    type (c_ptr) :: setPtr

  end type op_set

  type, BIND(C) :: op_map_core

    type(op_set_core) ::      from  ! set map from
    type(op_set_core) ::      to    ! set map to
    integer(kind=c_int) ::    dim   ! dimension of map
    type(c_ptr) ::            map   ! array defining map
    type(c_ptr) ::            name  ! map name

  end type op_map_core

  type op_map

    type(c_ptr) :: mapPtr

  end type op_map

  type, BIND(C) :: op_dat_core

    type(op_set_core) ::      set   ! set on which data is defined
    integer(kind=c_int) ::    dim   ! dimension of data
    integer(kind=c_int) ::    size  ! size of each element in dataset
    type(c_ptr) ::            dat   ! data on host
    type(c_ptr) ::            dat_d ! data on device
    type(c_ptr) ::            type  ! data type
    type(c_ptr) ::            name  ! data name

  end type op_dat_core

  type op_dat

    type(c_ptr) :: datPtr

  end type op_dat

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

    type(c_ptr) function op_decl_set_f ( setsize, name ) BIND(C,name='op_decl_set')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core

      integer(kind=c_int), value, intent(in)    :: setsize
      !type(op_set)                             :: set
      character(kind=c_char,len=1), intent(in)  :: name

    end function op_decl_set_f

    type(c_ptr) function op_decl_map_f ( from, to, mapdim, data, name ) BIND(C,name='op_decl_map_f')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core

      type(c_ptr), intent(in)          :: from, to
      integer(kind=c_int), value, intent(in) :: mapdim
      type(c_ptr), intent(in)                :: data
      character(kind=c_char,len=1)           :: name

    end function op_decl_map_f

    type(c_ptr) function op_decl_null_map () BIND(C,name='op_decl_null_map')

      use, intrinsic :: ISO_C_BINDING

    end function op_decl_null_map

    type(c_ptr) function op_decl_dat_f ( set, datdim, type, datsize, dat, name ) BIND(C,name='op_decl_dat_f')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_dat_core

      type(c_ptr), intent(in)            :: set
      integer(kind=c_int), value               :: datdim, datsize
      character(kind=c_char,len=1), intent(in) :: type
      type(c_ptr), intent(in)                  :: dat
      character(kind=c_char,len=1), intent(in) :: name

    end function op_decl_dat_f

    subroutine op_decl_const_f ( constdim, dat, name ) BIND(C,name='op_decl_const_f')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value :: constdim
      type(c_ptr), intent(in) :: dat
      character(kind=c_char,len=1) :: name

    end subroutine op_decl_const_f

    ! debug C functions (to obtain similar output file that can be diff-ed
    subroutine op_par_loop_2_f ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1  &
                             & ) BIND(C,name='op_par_loop_2')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(c_ptr) :: set
      type(c_ptr) :: data0, data1
      integer(kind=c_int), value :: itemSel0, itemSel1, access0, access1
      type(c_ptr) :: map0, map1

    end subroutine op_par_loop_2_f

    subroutine op_par_loop_5_F ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1, &
                               & data2, itemSel2, map2, access2, &
                               & data3, itemSel3, map3, access3, &
                               & data4, itemSel4, map4, access4 &
                             & ) BIND(C,name='op_par_loop_5')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

!     type(c_funptr) :: subroutineName
      type(c_ptr) :: set
      type(c_ptr) :: data0, data1, data2, data3, data4
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4
      integer(kind=c_int), value :: access0, access1, access2, access3, access4
      type(c_ptr) :: map0, map1, map2, map3, map4

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

      import :: op_set_core, op_map_core, op_dat_core

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

!     type(c_funptr) :: subroutineName
      type(c_ptr) :: set
      type(c_ptr) :: data0, data1, data2, data3, data4, data5
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5
      integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5
      type(c_ptr) :: map0, map1, map2, map3, map4, map5

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

      import :: op_set_core, op_map_core, op_dat_core

!     external subroutineName
      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

!     type(c_funptr) :: subroutineName
      type(c_ptr) :: set
      type(c_ptr) :: data0, data1, data2, data3, data4, data5, data6, data7
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5, itemSel6, itemSel7
      integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5, access6, access7
      type(c_ptr) :: map0, map1, map2, map3, map4, map5, map6, map7

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

!   subroutine op_fetchdata_C ( data ) BIND(C,name='op_fetch_data')
!
!     use, intrinsic :: ISO_C_BINDING
!
!     import :: op_dat_core
!
!     type(op_dat_core) :: data
!
!   end subroutine op_fetchdata_C


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
  end interface op_decl_gbl

  interface op_decl_const
    module procedure op_decl_const_integer_4, op_decl_const_real_8, op_decl_const_scalar_integer_4, &
                   & op_decl_const_scalar_real_8
  end interface op_decl_const

contains

  subroutine op_init ( diags )

    integer(4) :: diags

    integer(4) :: argc = 0

    type(c_ptr) :: opIDCPtr = C_NULL_PTR
    type(c_ptr) :: opGBLCPtr = C_NULL_PTR

    type (op_map_core), pointer :: idPtr
    type (op_map_core), pointer :: gblPtr

!   idPtr%dim = 0
!   gblPtr%dim = -1

!   type (op_map) :: actualopid

    OP_ID%mapPtr = op_decl_null_map ()
    OP_GBL%mapPtr = op_decl_null_map ()


!   call c_f_pointer ( opIDCPtr, OP_ID%mapPtr )
!   call c_f_pointer ( opGBLCPtr, OP_GBL%mapPtr )

!
!   call c_f_pointer ( opIDCPtr,  )
!   call c_f_pointer ( opGBLCPtr, gblPtr )
!
!   idPtr%dim = 0 ! OP_ID code used in arg_set
!   gblPtr%dim = -1 ! OP_GBL code used in arg_set

!   OP_ID = actualopid

!   OP_ID%mapPtr => idPtr
!   OP_GBL%mapPtr => gblPtr
!
    call op_init_core ( argc, C_NULL_PTR, diags )

  end subroutine op_init

  subroutine op_decl_set ( setsize, set, opname )

    integer(kind=c_int), value, intent(in) :: setsize
    type(op_set) :: set
    character(len=*), optional :: opname

    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

!   type(c_ptr) :: setCPtr = C_NULL_PTR
!   type(op_set_core), pointer :: setFPtr

    if ( present ( opname ) .eqv. .false. ) then
      set%setPtr = op_decl_set_F ( setsize, fakeName )
    else
      cname = C_CHAR_''//opname//C_NULL_CHAR
      set%setPtr = op_decl_set_F ( setsize, cname )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
!   call c_f_pointer ( setCPtr, set%setPtr )

  end subroutine op_decl_set

  subroutine op_decl_map ( from, to, mapdim, dat, map, opname )

    type(op_set), intent(in) :: from, to
    integer, intent(in) :: mapdim
    integer(4), dimension(*), intent(in), target :: dat
    type(op_map) :: map
    character(len=*), optional :: opname

    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

!   type(c_ptr) :: mapCPtr = C_NULL_PTR

    if ( present ( opname ) .eqv. .false. ) then
      map%mapPtr = op_decl_map_F ( from%setPtr, to%setPtr, mapdim, c_loc ( dat ), fakeName )
    else
      cname = C_CHAR_''//opname//C_NULL_CHAR
      map%mapPtr = op_decl_map_F ( from%setPtr, to%setPtr, mapdim, c_loc ( dat ), cname )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
!   call c_f_pointer ( mapCPtr, map%mapPtr )

  end subroutine op_decl_map

  subroutine op_decl_dat_real_8 ( set, datdim, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(len=*), optional :: opname

    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
    character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

!   type(c_ptr) :: dataCPtr = C_NULL_PTR

    if ( present ( opname ) .eqv. .false. ) then
      data%datPtr = op_decl_dat_f ( set%setPtr, datdim, type, 8, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      data%datPtr = op_decl_dat_f ( set%setPtr, datdim, type, 8, c_loc ( dat ), cname )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
!   call c_f_pointer ( dataCPtr, data%datPtr )

  end subroutine op_decl_dat_real_8

  subroutine op_decl_dat_integer_4 ( set, datdim, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(len=*), optional :: opname

    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR
    character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

!   type(c_ptr) :: dataCPtr = C_NULL_PTR

    if ( present ( opname ) .eqv. .false. ) then
      data%datPtr = op_decl_dat_f ( set%setPtr, datdim, type, 4, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      data%datPtr = op_decl_dat_f ( set%setPtr, datdim, type, 4, c_loc ( dat ), cname )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
!   call c_f_pointer ( dataCPtr, data%datPtr )

  end subroutine op_decl_dat_integer_4

  subroutine op_decl_gbl_real_8 ( dat, gbldata, gbldim )

    real(8), dimension(*), intent(in), target :: dat
    type(op_dat) :: gblData
    integer, intent(in) :: gbldim

    ! unused name
    character(kind=c_char,len=7) :: name = C_CHAR_'NONAME'//C_NULL_CHAR
    character(kind=c_char,len=5) :: type = C_CHAR_'real'//C_NULL_CHAR

    ! unsed op_set
    type(c_ptr) :: unusedSet = C_NULL_PTR

    type(c_ptr) :: gblCPtr = C_NULL_PTR

    gblData%datPtr = op_decl_dat_f ( unusedSet, gbldim, type, 8, c_loc ( dat ), name )

!   call c_f_pointer ( gblCPtr, gblData%datPtr )

  end subroutine op_decl_gbl_real_8

  subroutine op_decl_const_integer_4 ( constdim, dat, opname )

    integer(kind=c_int), value :: constdim
    integer(4), dimension(*), intent(in), target :: dat
    character(len=*), optional :: opname
    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

    if ( present ( opname ) .eqv. .false. ) then
      call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      call op_decl_const_F ( constdim, c_loc ( dat ), cname )
    end if

  end subroutine op_decl_const_integer_4

  subroutine op_decl_const_real_8 ( constdim, dat, opname )

    integer(kind=c_int), value :: constdim
    real(8), dimension(*), intent(in), target :: dat
    character(len=*), optional :: opname
    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

    if ( present ( opname ) .eqv. .false. ) then
      call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      call op_decl_const_F ( constdim, c_loc ( dat ), cname )
    end if

  end subroutine op_decl_const_real_8

  subroutine op_decl_const_scalar_integer_4 ( constdim, dat, opname )

    integer(kind=c_int), value :: constdim
    integer(4), intent(in), target :: dat
    character(len=*), optional :: opname
    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

    if ( present ( opname ) .eqv. .false. ) then
      call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      call op_decl_const_F ( constdim, c_loc ( dat ), cname )
    end if

  end subroutine op_decl_const_scalar_integer_4

  subroutine op_decl_const_scalar_real_8 ( constdim, dat, opname )

    integer(kind=c_int), value :: constdim
    real(8), intent(in), target :: dat
    character(len=*), optional :: opname
    character(len=len(opname)+1) :: cname

    character(kind=c_char,len=7) :: fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

    if ( present ( opname ) .eqv. .false. ) then
      call op_decl_const_F ( constdim, c_loc ( dat ), fakeName )
    else
      cname = opname//char(0)
      call op_decl_const_F ( constdim, c_loc ( dat ), cname )
    end if

  end subroutine op_decl_const_scalar_real_8

! subroutine op_fetchdata ( data )
!
!   type(op_dat) :: data
!
!   call op_fetchdata_C ( data%datPtr )
!
! end subroutine op_fetchdata


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
    call op_par_loop_2_f ( subroutineName, set%setPtr, &
                         & data0%datPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%datPtr, itemSelC1, map1%mapPtr, access1 &
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
    call op_par_loop_5_f ( subroutineName, set%setPtr, &
                         & data0%datPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%datPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%datPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%datPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%datPtr, itemSelC4, map4%mapPtr, access4  &
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
    call op_par_loop_6_f ( subroutineName, set%setPtr, &
                         & data0%datPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%datPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%datPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%datPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%datPtr, itemSelC4, map4%mapPtr, access4, &
                         & data5%datPtr, itemSelC5, map5%mapPtr, access5  &
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
!     type(c_funptr) :: subroutineName
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
    call op_par_loop_8_f ( subroutineName, set%setPtr, &
                         & data0%datPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%datPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%datPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%datPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%datPtr, itemSelC4, map4%mapPtr, access4, &
                         & data5%datPtr, itemSelC5, map5%mapPtr, access5, &
                         & data6%datPtr, itemSelC6, map6%mapPtr, access6, &
                         & data7%datPtr, itemSelC7, map7%mapPtr, access7  &
                       & )

    end subroutine op_par_loop_8

end module OP2_Fortran

