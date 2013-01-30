!
! Open source copyright declaration based on BSD open source template:
! http://www.opensource.org/licenses/bsd-license.php
!
! This file is part of the OP2 distribution.
!
! Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
! the main source directory for a full list of copyright holders.
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!     * Redistributions of source code must retain the above copyright
!       notice, this list of conditions and the following disclaimer.
!     * Redistributions in binary form must reproduce the above copyright
!       notice, this list of conditions and the following disclaimer in the
!       documentation and/or other materials provided with the distribution.
!     * The name of Mike Giles may not be used to endorse or promote products
!       derived from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
! DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
! (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
! ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!

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
  integer(c_int) :: OP_RW = 3
  integer(c_int) :: OP_INC = 4
  integer(c_int) :: OP_MIN = 5
  integer(c_int) :: OP_MAX = 6

  type, BIND(C) :: op_set_core

   integer(kind=c_int) :: index        ! position in the private OP2 array of op_set_core variables
    integer(kind=c_int) :: size         ! number of elements in the set
    type(c_ptr)         :: name         ! set name
    integer(kind=c_int) :: core_size    ! number of core elements in an mpi process
    integer(kind=c_int) :: exec_size    ! number of additional imported elements to be executed
    integer(kind=c_int) :: nonexec_size ! number of additional imported elements that are not executed

  end type op_set_core

  type :: op_set

    type (op_set_core), pointer :: setPtr => null()
    type(c_ptr)                 :: setCptr

  end type op_set

  type, BIND(C) :: op_map_core

    integer(kind=c_int) ::    index        ! position in the private OP2 array of op_map_core variables
    type(c_ptr) ::            from         ! set map from
    type(c_ptr) ::            to           ! set map to
    integer(kind=c_int) ::    dim          ! dimension of map
    type(c_ptr) ::            map          ! array defining map
    type(c_ptr) ::            name         ! map name
    integer(kind=c_int) ::    user_managed ! indicates whether the user is managing memory

  end type op_map_core

  type :: op_map

    type(op_map_core), pointer :: mapPtr => null()
    type(c_ptr) :: mapCptr

  end type op_map

  type, BIND(C) :: op_dat_core

    integer(kind=c_int) ::    index        ! position in the private OP2 array of op_dat_core variables
    type(c_ptr) ::            set          ! set on which data is defined
    integer(kind=c_int) ::    dim          ! dimension of data
    integer(kind=c_int) ::    size         ! size of each element in dataset
    type(c_ptr) ::            dat          ! data on host
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::         dat_d        ! data on device
#else
    type(c_ptr) ::            dat_d        ! data on device
#endif
    type(c_ptr) ::            type         ! data type
    type(c_ptr) ::            name         ! data name
    type(c_ptr) ::            buffer_d     ! buffer for MPI halo sends on the device
    integer(kind=c_int) ::    dirtybit     ! flag to indicate MPI halo exchange is needed
    integer(kind=c_int) ::    user_managed ! indicates whether the user is managing memory

  end type op_dat_core

  type op_dat

    type(op_dat_core), pointer :: dataPtr => null()
    type(c_ptr) :: dataCptr

  end type op_dat

  type, BIND(C) :: op_arg

    integer(kind=c_int) :: index
    type(c_ptr)         :: dat
    type(c_ptr)         :: map
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: idx
    integer(kind=c_int) :: size
    type(c_ptr)         :: data
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::         data_d    ! data on device
#else
    type(c_ptr)         :: data_d
#endif
    type(c_ptr)         :: type
    integer(kind=c_int) :: acc
    integer(kind=c_int) :: argtype
    integer(kind=c_int) :: sent

  end type op_arg

  ! declaration of identity and global mapping
  type(op_map) :: OP_ID, OP_GBL

  ! Declarations of op_par_loop implemented in C
  interface

    subroutine op_init_c ( argc, argv, diags ) BIND(C,name='op_init')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags

    end subroutine op_init_c

    subroutine op_exit_c (  ) BIND(C,name='op_exit')

      use, intrinsic :: ISO_C_BINDING

    end subroutine op_exit_c

    type(c_ptr) function op_decl_set_c ( setsize, name ) BIND(C,name='op_decl_set')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core

      integer(kind=c_int), value, intent(in)    :: setsize
      character(kind=c_char,len=1), intent(in)  :: name(*)

    end function op_decl_set_c

    INTEGER function op_get_size_c ( set ) BIND(C,name='op_get_size')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function



    type(c_ptr) function op_decl_map_c ( from, to, mapdim, data, name ) BIND(C,name='op_decl_map')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: from, to
      integer(kind=c_int), value, intent(in)   :: mapdim
      type(c_ptr), intent(in), value           :: data
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function op_decl_map_c

    type(c_ptr) function op_decl_null_map () BIND(C,name='op_decl_null_map')

      use, intrinsic :: ISO_C_BINDING

    end function op_decl_null_map

    type(c_ptr) function op_decl_dat_c ( set, datdim, type, datsize, dat, name ) BIND(C,name='op_decl_dat_char')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_dat_core

      type(c_ptr), value, intent(in)           :: set
      integer(kind=c_int), value               :: datdim, datsize
      character(kind=c_char,len=1), intent(in) :: type(*)
      type(c_ptr), intent(in), value           :: dat
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function op_decl_dat_c

    function op_arg_dat_c ( dat, idx, map, dim, type, acc ) BIND(C,name='op_arg_dat')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_dat_c

      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
      integer(kind=c_int), value :: dim
      type(c_ptr), value :: type
      integer(kind=c_int), value :: acc

    end function op_arg_dat_c

    function op_arg_dat_null_c ( dat, idx, map, dim, type, acc ) BIND(C,name='op_arg_dat_null')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_dat_null_c

      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
      integer(kind=c_int), value :: dim
      type(c_ptr), value :: type
      integer(kind=c_int), value :: acc

    end function op_arg_dat_null_c


    function op_arg_gbl_c ( dat, dim, type, size, acc ) BIND(C,name='op_arg_gbl_copy')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_gbl_c

      type(c_ptr), value :: dat
      integer(kind=c_int), value :: dim
      character(kind=c_char), dimension(*) :: type
      integer(kind=c_int), value :: size
      integer(kind=c_int), value :: acc

    end function op_arg_gbl_c

    subroutine print_type (type) BIND(C,name='print_type')

      use, intrinsic :: ISO_C_BINDING
      import :: op_arg

      type(op_arg) :: type

    end subroutine

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

    subroutine op_get_dat_c ( opdat ) BIND(C,name='op_get_dat')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine

    subroutine op_put_dat_c ( opdat ) BIND(C,name='op_put_dat')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine

    subroutine op_get_dat_mpi_c ( opdat ) BIND(C,name='op_get_dat_mpi')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine op_get_dat_mpi_c

    subroutine op_put_dat_mpi_c ( opdat ) BIND(C,name='op_put_dat_mpi')

      import :: op_dat_core

      type(op_dat_core) :: opdat

    end subroutine op_put_dat_mpi_c

    ! this function shall be removed after debugging HYDRA (the above ones should be used instead)
!    subroutine op_fetch_data ( opdat ) BIND(C,name='op_fetch_data')

!      import :: op_dat_core

!      type(op_dat_core) :: opdat

!    end subroutine op_fetch_data

   subroutine dumpOpDatFromDevice_c ( data, label, sequenceNumber ) BIND(C,name='dumpOpDatFromDevice')
      use, intrinsic :: ISO_C_BINDING

     import :: op_dat_core

     type(op_dat_core) :: data
     character(len=1,kind=c_char) :: label(*)
     integer(kind=c_int) :: sequenceNumber

   end subroutine

   subroutine dumpOpDat_c ( data, fileName ) BIND(C,name='dumpOpDat')
      use, intrinsic :: ISO_C_BINDING

     import :: op_dat_core

     type(op_dat_core) :: data
     character(len=1,kind=c_char) :: fileName(*)

   end subroutine

   subroutine dumpOpMap_c ( map, fileName ) BIND(C,name='dumpOpMap')
     use, intrinsic :: ISO_C_BINDING

     import :: op_map_core

     type(op_map_core) :: map
     character(len=1,kind=c_char) :: fileName(*)

   end subroutine

   subroutine op_mpi_rank_c (rank) BIND(C,name='op_mpi_rank')

     use, intrinsic :: ISO_C_BINDING

     integer(kind=c_int) :: rank

   end subroutine op_mpi_rank_c

   subroutine printFirstDatPosition (data) BIND(C,name='printFirstDatPosition')
     import :: op_dat_core

     type(op_dat_core) :: data

   end subroutine printFirstDatPosition

    subroutine op_diagnostic_output (  ) BIND(C,name='op_diagnostic_output')
      use, intrinsic :: ISO_C_BINDING
    end subroutine

    subroutine op_print_dat_to_binfile_c (dat, fileName) BIND(C,name='op_print_dat_to_binfile')
      use, intrinsic :: ISO_C_BINDING

      import :: op_dat_core

      type(op_dat_core) :: dat
      character(len=1,kind=c_char) :: fileName(*)

    end subroutine op_print_dat_to_binfile_c

    logical(kind=c_bool) function isCNullPointer_c (ptr) BIND(C,name='isCNullPointer')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value :: ptr
    end function isCNullPointer_c

    subroutine op_timing_output () BIND(C,name='op_timing_output')

    end subroutine op_timing_output

  end interface

  ! the two numbers at the end of the name indicate the size of the type (e.g. real(8))
  interface op_decl_dat
    module procedure op_decl_dat_real_8, op_decl_dat_integer_4, &
                     op_decl_dat_real_8_2, op_decl_dat_integer_4_2, &
                     op_decl_dat_real_8_3, op_decl_dat_integer_4_3
  end interface op_decl_dat

  interface op_arg_gbl
    module procedure op_arg_gbl_real_8_scalar, op_arg_gbl_real_8, op_arg_gbl_real_8_2, &
                   & op_arg_gbl_integer_4_scalar, op_arg_gbl_integer_4, op_arg_gbl_integer_4_2, &
       & op_arg_gbl_logical_scalar, op_arg_gbl_logical, op_arg_gbl_python
  end interface op_arg_gbl

  interface op_decl_const
    module procedure op_decl_const_integer_4, op_decl_const_real_8, op_decl_const_scalar_integer_4, &
    & op_decl_const_scalar_real_8, op_decl_const_logical
  end interface op_decl_const

  interface op_arg_dat
    module procedure op_arg_dat_rose, op_arg_dat_python
  end interface op_arg_dat

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

    call op_init_c ( argc, C_NULL_PTR, diags )

  end subroutine op_init


  subroutine op_exit ( )

    call op_exit_c (  )

  end subroutine op_exit


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

    ! debugging

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
  !   declarations of constants  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! All of these are no-ops in the reference implementation

  subroutine op_decl_const_integer_4 ( dat, constdim, opname )

    integer(4), dimension(:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    ! local dummies to prevent compiler warning
    integer(4), dimension(1) :: dat_dummy
    integer(kind=c_int) :: constdim_dummy
    character(kind=c_char) :: opname_dummy

    dat_dummy = dat
    constdim_dummy = constdim
    opname_dummy = opname

  end subroutine op_decl_const_integer_4

  subroutine op_decl_const_real_8 ( dat, constdim, opname )

    real(8), dimension(:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    ! local dummies to prevent compiler warning
    real(8), dimension(1) :: dat_dummy
    integer(kind=c_int) :: constdim_dummy
    character(kind=c_char) :: opname_dummy

    dat_dummy = dat
    constdim_dummy = constdim
    opname_dummy = opname

  end subroutine op_decl_const_real_8

  subroutine op_decl_const_scalar_integer_4 ( dat, constdim, opname )

    integer(4), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    ! local dummies to prevent compiler warning
    integer(4) :: dat_dummy
    integer(kind=c_int) :: constdim_dummy
    character(kind=c_char) :: opname_dummy

    dat_dummy = dat
    constdim_dummy = constdim
    opname_dummy = opname

  end subroutine op_decl_const_scalar_integer_4

  subroutine op_decl_const_scalar_real_8 ( dat, constdim, opname )

    real(8), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    ! local dummies to prevent compiler warning
    real(8) :: dat_dummy
    integer(kind=c_int) :: constdim_dummy
    character(kind=c_char) :: opname_dummy

    dat_dummy = dat
    constdim_dummy = constdim
    opname_dummy = opname

  end subroutine op_decl_const_scalar_real_8

  subroutine op_decl_const_logical ( dat, constdim, opname )

    logical, intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    ! local dummies to prevent compiler warning
    logical :: dat_dummy
    integer(kind=c_int) :: constdim_dummy
    character(kind=c_char) :: opname_dummy

    dat_dummy = dat
    constdim_dummy = constdim
    opname_dummy = opname

  end subroutine op_decl_const_logical

  type(op_arg) function op_arg_dat_rose (dat, idx, map, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_dat) :: dat
    integer(kind=c_int) :: idx
    type(op_map) :: map
    integer(kind=c_int) :: access

    ! first check if the op_dat is actually declared (HYDRA feature)
    ! If is NULL, then return an empty op_arg
#ifdef OP2_WITH_CUDAFOR
    if (dat%dataCPtr .eq. C_NULL_PTR) then
#else
    if ( isCNullPointer_c (dat%dataCPtr) .eqv. .true. ) then
#endif
      op_arg_dat_rose = op_arg_dat_null_c (C_NULL_PTR, idx-1, C_NULL_PTR, -1, C_NULL_PTR, access-1)
    else
      ! warning: access and idx are in FORTRAN style, while the C style is required here
      if ( map%mapPtr%dim .eq. 0 ) then
        ! OP_ID case (does not decrement idx)
        op_arg_dat_rose = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR, dat%dataPtr%dim, dat%dataPtr%type, access-1 )
      else
        op_arg_dat_rose = op_arg_dat_c ( dat%dataCPtr, idx-1, map%mapCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )
      endif
    endif

  end function op_arg_dat_rose

  type(op_arg) function op_arg_dat_python (dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_dat) :: dat
    integer(kind=c_int) :: idx
    type(op_map) :: map
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access

    ! first check if the op_dat is actually declared (HYDRA feature)
    ! If is NULL, then return an empty op_arg
#ifdef OP2_WITH_CUDAFOR
    if (dat%dataCPtr .eq. C_NULL_PTR) then
#else
    if ( isCNullPointer_c (dat%dataCPtr) .eqv. .true. ) then
#endif
      op_arg_dat_python = op_arg_dat_null_c (C_NULL_PTR, idx-1, C_NULL_PTR, -1, C_NULL_PTR, access-1)
    else
      ! warning: access and idx are in FORTRAN style, while the C style is required here
      if ( map%mapPtr%dim .eq. 0 ) then
        ! OP_ID case (does not decrement idx)
        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR,  dat%dataPtr%dim, dat%dataPtr%type, access-1 )
      else
        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx-1, map%mapCPtr,  dat%dataPtr%dim, dat%dataPtr%type, access-1 )
      endif
    endif

  end function op_arg_dat_python

  type(op_arg) function op_arg_dat_generic (dat, idx, map, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_dat) :: dat
    integer(kind=c_int) :: idx
    type(op_map) :: map
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access

!    dim = dim
!    type = type

    ! warning: access and idx are in FORTRAN style, while the C style is required here
    if ( map%mapPtr%dim .eq. 0 ) then
      ! OP_ID case (does not decrement idx)
      op_arg_dat_generic = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR, dat%dataPtr%dim, dat%dataPtr%type, access-1 )
    else
      op_arg_dat_generic = op_arg_dat_c ( dat%dataCPtr, idx-1, map%mapCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )
    endif

  end function op_arg_dat_generic


    INTEGER function op_get_size (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size = op_get_size_c ( set%setCPtr )

  end function op_get_size







  type(op_arg) function op_arg_gbl_real_8_scalar ( dat, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), target :: dat
    integer(kind=c_int) :: access

    character(kind=c_char,len=7) :: type = C_CHAR_'double'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_real_8_scalar = op_arg_gbl_c ( c_loc (dat), 1, type, 8, access-1 )

  end function op_arg_gbl_real_8_scalar

  type(op_arg) function op_arg_gbl_real_8 ( dat, dim, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access

    character(kind=c_char,len=7) :: type = C_CHAR_'double'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_real_8 = op_arg_gbl_c ( c_loc (dat), dim, type, 8, access-1 )

  end function op_arg_gbl_real_8

  type(op_arg) function op_arg_gbl_python ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'double'//C_NULL_CHAR, 8, access-1 )
    !op_arg_dat_python = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python




  type(op_arg) function op_arg_gbl_real_8_2 ( dat, dim, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), dimension(:,:) :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access

    op_arg_gbl_real_8_2 = op_arg_gbl_real_8 ( dat, dim, access )

  end function op_arg_gbl_real_8_2

  type(op_arg) function op_arg_gbl_integer_4_scalar ( dat, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), target :: dat
    integer(kind=c_int) :: access

    character(kind=c_char,len=4) :: type = C_CHAR_'int'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_integer_4_scalar = op_arg_gbl_c ( c_loc (dat), 1, type, 4, access-1 )

  end function op_arg_gbl_integer_4_scalar

  type(op_arg) function op_arg_gbl_integer_4 ( dat, dim, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access

    character(kind=c_char,len=4) :: type = C_CHAR_'int'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_integer_4 = op_arg_gbl_c ( c_loc (dat), dim, type, 4, access-1 )

  end function op_arg_gbl_integer_4

  type(op_arg) function op_arg_gbl_integer_4_2 ( dat, dim, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), dimension(:,:) :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access

    op_arg_gbl_integer_4_2 = op_arg_gbl_integer_4 ( dat, dim, access )

  end function op_arg_gbl_integer_4_2


  type(op_arg) function op_arg_gbl_logical_scalar ( dat, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical, target :: dat
    integer(kind=c_int) :: access

    character(kind=c_char,len=5) :: type = C_CHAR_'bool'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_logical_scalar = op_arg_gbl_c ( c_loc (dat), 1, type, 1, access-1 )

  end function op_arg_gbl_logical_scalar

  type(op_arg) function op_arg_gbl_logical ( dat, dim, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical, dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access

    character(kind=c_char,len=5) :: type = C_CHAR_'bool'//C_NULL_CHAR

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_logical = op_arg_gbl_c ( c_loc (dat(1)), dim, type, 1, access-1 )

  end function op_arg_gbl_logical

  subroutine op_get_dat ( opdat )

    type(op_dat) :: opdat

    call op_get_dat_c ( opdat%dataPtr)

  end subroutine op_get_dat

  subroutine op_put_dat ( opdat )

    type(op_dat) :: opdat

    call op_put_dat_c ( opdat%dataPtr)

  end subroutine op_put_dat

  subroutine op_get_dat_mpi ( opdat )

    type(op_dat) :: opdat

    call op_get_dat_mpi_c ( opdat%dataPtr)

  end subroutine op_get_dat_mpi

  subroutine op_put_dat_mpi ( opdat )

    type(op_dat) :: opdat

    call op_put_dat_mpi_c ( opdat%dataPtr)

  end subroutine op_put_dat_mpi

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

  subroutine op_print_dat_to_binfile (dat, fileName)

    type(op_dat) :: dat
    character(len=*) :: fileName

    call op_print_dat_to_binfile_c (dat%dataPtr, fileName)

  end subroutine op_print_dat_to_binfile

  subroutine op_mpi_rank (rank)

    integer(kind=c_int) :: rank

    call op_mpi_rank_c (rank)

  end subroutine op_mpi_rank

end module OP2_Fortran_Declarations
