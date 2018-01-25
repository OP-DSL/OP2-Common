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
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::         map_d        ! array defining map on device
#else
    type(c_ptr) ::            map_d        ! array defining map on device
#endif
    type(c_ptr) ::            name         ! map name
    integer(kind=c_int) ::    user_managed ! indicates whether the user is managing memory

  end type op_map_core

  type :: op_map

    type(op_map_core), pointer :: mapPtr => null()
    type(c_ptr) :: mapCptr
    integer (kind=c_int) :: status = -1

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
    integer(kind=c_int) ::    dirty_hd     ! flag to indicate dirty status on host and device
    integer(kind=c_int) ::    user_managed ! indicates whether the user is managing memory

  end type op_dat_core

  type op_dat

    type(op_dat_core), pointer :: dataPtr => null()
    type(c_ptr) :: dataCptr
    integer (kind=c_int) :: status = -1

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
    type(c_ptr)         :: map_data
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::         map_data_d    ! data on device
#else
    type(c_ptr)         :: map_data_d
#endif
    type(c_ptr)         :: type
    integer(kind=c_int) :: acc
    integer(kind=c_int) :: argtype
    integer(kind=c_int) :: sent
    integer(kind=c_int) :: opt

  end type op_arg

  ! declaration of identity and global mapping
  type(op_map) :: OP_ID, OP_GBL

  type, BIND(C) :: op_export_core

    integer(kind=c_int) :: index
    integer(kind=c_int) :: coupling_group_size
    type(c_ptr)         :: coupling_proclist

    integer(kind=c_int) :: num_ifaces
    type(c_ptr)         :: iface_list

    type(c_ptr)         :: nprocs_per_int
    type(c_ptr)         :: proclist_per_int
    type(c_ptr)         :: nodelist_send_size
    type(c_ptr)         :: nodelist_send

    integer(kind=c_int) :: max_data_size
    type(c_ptr)         :: send_buf
    type(c_ptr)         :: requests
    type(c_ptr)         :: statuses

    type(c_ptr)         :: OP_global_buffer
    integer(kind=c_int) :: OP_global_buffer_size

    integer(kind=c_int) :: gbl_num_ifaces
    type(c_ptr)         :: gbl_iface_list
    type(c_ptr)         :: nprocs_per_gint
    type(c_ptr)         :: proclist_per_gint

    integer(kind=c_int) :: gbl_offset
    type(c_ptr)         :: cellsToNodes
    type(c_ptr)         :: coords
    type(c_ptr)         :: mark

  end type op_export_core

  type :: op_export_handle

    type(op_export_core), pointer :: exportPtr => null()
    type(c_ptr)                   :: exportCptr
    integer(kind=c_int)           :: status = -1

  end type op_export_handle


  type, BIND(C) :: op_import_core

    integer(kind=c_int)  :: index
    integer(kind=c_int)  :: nprocs
    type(c_ptr)          :: proclist
    integer(kind=c_int)  :: gbl_offset
    type(c_ptr)          :: coords
    type(c_ptr)          :: mark
    integer(kind=c_int)  :: max_dat_size
    integer(kind=c_int)  :: num_my_ifaces
    type(c_ptr)          :: iface_list
    type(c_ptr)          :: nprocs_per_int
    type(c_ptr)          :: proclist_per_int
    type(c_ptr)          :: node_size_per_int
    type(c_ptr)          :: nodelist_per_int
    type(c_ptr)          :: recv_buf
    type(c_ptr)          :: recv2int
    type(c_ptr)          :: recv2proc
    type(c_ptr)          :: requests
    type(c_ptr)          :: statuses
    type(c_ptr)          :: interp_dist

  end type op_import_core

  type :: op_import_handle

    type(op_import_core), pointer :: importPtr => null()
    type(c_ptr)                   :: importCptr
    integer(kind=c_int)           :: status = -1

  end type op_import_handle

  ! Declarations of op_par_loop implemented in C
  interface

    subroutine op_init_c ( argc, argv, diags ) BIND(C,name='op_init')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags

    end subroutine op_init_c

    subroutine op_init_soa_c ( argc, argv, diags, soa ) BIND(C,name='op_init_soa')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags
      integer(kind=c_int), intent(in), value :: soa

    end subroutine op_init_soa_c

    subroutine op_set_args_c ( argc, argv ) BIND(C,name='op_set_args')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: argc
      character(len=1, kind=C_CHAR) :: argv
    end subroutine op_set_args_c

    subroutine op_mpi_init_c ( argc, argv, diags, global, local ) BIND(C,name='op_mpi_init')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags
      integer(kind=c_int), intent(in), value :: global
      integer(kind=c_int), intent(in), value :: local
    end subroutine op_mpi_init_c

    subroutine op_mpi_init_soa_c ( argc, argv, diags, global, local, soa ) BIND(C,name='op_mpi_init_soa')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags
      integer(kind=c_int), intent(in), value :: global
      integer(kind=c_int), intent(in), value :: local
      integer(kind=c_int), intent(in), value :: soa
    end subroutine op_mpi_init_soa_c

    subroutine op_exit_c (  ) BIND(C,name='op_exit')

      use, intrinsic :: ISO_C_BINDING

    end subroutine op_exit_c

    subroutine op_decl_const_char_c ( dim, type, typesize, data, name ) BIND(C,name='op_decl_const_char')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: dim, typesize
      character(kind=c_char,len=1), intent(in)  :: type(*)
      type(c_ptr), value, intent(in)                :: data
      character(kind=c_char,len=1), intent(in)  :: name(*)
    end subroutine op_decl_const_char_c

    type(c_ptr) function op_decl_set_c ( setsize, name ) BIND(C,name='op_decl_set')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core

      integer(kind=c_int), value, intent(in)    :: setsize
      character(kind=c_char,len=1), intent(in)  :: name(*)

    end function op_decl_set_c

    INTEGER(kind=c_int) function op_get_size_c ( set ) BIND(C,name='op_get_size')
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
      character(kind=c_char,len=1) :: type(*)
      integer(kind=c_int), value :: acc

    end function op_arg_dat_c

    function op_opt_arg_dat_c ( opt, dat, idx, map, dim, type, acc ) BIND(C,name='op_opt_arg_dat')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_opt_arg_dat_c

      integer(kind=c_int), value :: opt
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1) :: type(*)
      integer(kind=c_int), value :: acc

    end function op_opt_arg_dat_c

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

    subroutine op_fetch_data_c ( opdat, data ) BIND(C,name='op_fetch_data_char')
      use, intrinsic :: ISO_C_BINDING
      import :: op_dat_core

      type(op_dat_core) :: opdat
      type(c_ptr), value :: data

    end subroutine op_fetch_data_c

    subroutine op_fetch_data_idx_c ( opdat, data, low, high) BIND(C,name='op_fetch_data_idx_char')
      use, intrinsic :: ISO_C_BINDING
      import :: op_dat_core

      type(op_dat_core) :: opdat
      type(c_ptr), value :: data
      integer(kind=c_int), value :: high
      integer(kind=c_int), value :: low

    end subroutine op_fetch_data_idx_c

    subroutine op_timers_core_f ( cpu, et ) BIND(C,name='op_timers_core')
      use, intrinsic :: ISO_C_BINDING

      real(kind=c_double) :: cpu, et

    end subroutine op_timers_core_f

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

    subroutine op_generate_consts_header (  ) BIND(C,name='op_generate_consts_header')
      use, intrinsic :: ISO_C_BINDING
    end subroutine

    subroutine op_print_dat_to_binfile_c (dat, fileName) BIND(C,name='op_print_dat_to_binfile')
      use, intrinsic :: ISO_C_BINDING

      import :: op_dat_core

      type(op_dat_core) :: dat
      character(len=1,kind=c_char) :: fileName(*)

    end subroutine op_print_dat_to_binfile_c

    subroutine op_print_dat_to_txtfile_c (dat, fileName) BIND(C,name='op_print_dat_to_txtfile')
      use, intrinsic :: ISO_C_BINDING

      import :: op_dat_core

      type(op_dat_core) :: dat
      character(len=1,kind=c_char) :: fileName(*)

    end subroutine op_print_dat_to_txtfile_c

    logical(kind=c_bool) function isCNullPointer_c (ptr) BIND(C,name='isCNullPointer')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value :: ptr
    end function isCNullPointer_c

    subroutine op_timing_output () BIND(C,name='op_timing_output')

    end subroutine op_timing_output

    subroutine op_print_c (line) BIND(C,name='op_print')
      use ISO_C_BINDING

      character(kind=c_char) :: line(*)
    end subroutine op_print_c

    type (c_ptr) function op_import_init_size_c (nprocs, proclist_ptr, mark) BIND(C,name='op_import_init_size')
      use ISO_C_BINDING

      import :: op_dat_core

      integer(c_int), value :: nprocs
      type(c_ptr), value    :: proclist_ptr
      type(op_dat_core)     :: mark

    end function op_import_init_size_c

    type (c_ptr) function op_import_init_c (exp_handle, coords, mark) BIND(C,name='op_import_init')
      use ISO_C_BINDING

      import :: op_dat_core
      import :: op_export_core

      type(op_export_core)  :: exp_handle
      type(op_dat_core)     :: coords
      type(op_dat_core)     :: mark

    end function op_import_init_c

    type (c_ptr) function op_export_init_c (nprocs, proclist_ptr, cells2Nodes, sp_nodes, coords, mark) BIND(C,name='op_export_init')
      use ISO_C_BINDING

      import :: op_dat_core
      import :: op_map_core
      import :: op_set_core

      integer(c_int), value :: nprocs
      type(c_ptr), value    :: proclist_ptr
      type(op_map_core)     :: cells2Nodes
      type(op_set_core)     :: sp_nodes
      type(op_dat_core)     :: coords
      type(op_dat_core)     :: mark

    end function op_export_init_c

    subroutine op_export_data_c (exp_handle, dat) BIND(C,name='op_export_data')
      use ISO_C_BINDING

      import :: op_export_core
      import :: op_dat_core

      type(op_export_core)  :: exp_handle
      type(op_dat_core)     :: dat

    end subroutine op_export_data_c

    subroutine op_import_data_c (imp_handle, dat) BIND(C,name='op_import_data')
      use ISO_C_BINDING

      import :: op_import_core
      import :: op_dat_core

      type(op_import_core)  :: imp_handle
      type(op_dat_core)     :: dat

    end subroutine op_import_data_c

    subroutine op_inc_theta_c (exp_handle, bc_id, dtheta_exp, dtheta_imp) BIND(C,name='op_inc_theta')
      use ISO_C_BINDING

      import :: op_export_core

      type(op_export_core)  :: exp_handle
      type(c_ptr), value    :: bc_id
      type(c_ptr), value    :: dtheta_exp
      type(c_ptr), value    :: dtheta_imp

    end subroutine op_inc_theta_c

    subroutine op_theta_init_c (exp_handle, bc_id, dtheta_exp, dtheta_imp, alpha) BIND(C,name='op_theta_init')
      use ISO_C_BINDING

      import :: op_export_core

      type(op_export_core)  :: exp_handle
      type(c_ptr), value    :: bc_id
      type(c_ptr), value    :: dtheta_exp
      type(c_ptr), value    :: dtheta_imp
      type(c_ptr), value    :: alpha

    end subroutine op_theta_init_c

    subroutine set_maps_base_c (base) BIND(C,name='set_maps_base')
      use ISO_C_BINDING
      integer(c_int), value :: base
    end subroutine set_maps_base_c


  end interface

  ! the two numbers at the end of the name indicate the size of the type (e.g. real(8))
  interface op_decl_dat
    module procedure op_decl_dat_real_8, op_decl_dat_integer_4, &
                     op_decl_dat_real_8_2, op_decl_dat_integer_4_2, &
                     op_decl_dat_real_8_3, op_decl_dat_integer_4_3
  end interface op_decl_dat

  interface op_arg_gbl
    module procedure op_arg_gbl_python_r8_scalar, &
       & op_arg_gbl_python_i4_scalar, op_arg_gbl_python_logical_scalar, op_arg_gbl_python_r8_1dim, &
       & op_arg_gbl_python_i4_1dim, op_arg_gbl_python_logical_1dim, op_arg_gbl_python_r8_2dim, &
       & op_arg_gbl_python_i4_2dim, op_arg_gbl_python_logical_2dim, &
       & op_arg_gbl_python_r8_3dim
  end interface op_arg_gbl

  interface op_decl_const
    module procedure op_decl_const_integer_4, op_decl_const_real_8, op_decl_const_scalar_integer_4, &
    & op_decl_const_scalar_real_8, op_decl_const_logical, op_decl_const_logical_1d, &
    & op_decl_const_integer_2_4, op_decl_const_real_2_8
  end interface op_decl_const

  interface op_arg_dat
    module procedure op_arg_dat_python
  end interface op_arg_dat

  interface op_opt_arg_dat
    module procedure op_opt_arg_dat_python
  end interface op_opt_arg_dat

  interface op_fetch_data
    module procedure op_fetch_data_real_8, op_fetch_data_real_4, &
    op_fetch_data_integer_4
  end interface op_fetch_data

  interface op_fetch_data_idx
    module procedure op_fetch_data_idx_real_8, op_fetch_data_idx_real_4, &
    op_fetch_data_idx_integer_4
  end interface op_fetch_data_idx

contains

  subroutine op_init_base_soa ( diags, base_idx, soa )

    ! formal parameter
    integer(4) :: diags
    integer(4) :: base_idx
    integer(4) :: soa

    ! local variables
    integer(4) :: argc = 0
    integer :: i
    character(kind=c_char,len=64)           :: temp

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
    call set_maps_base_c(base_idx)

    call op_init_soa_c ( 0, C_NULL_PTR, diags, soa )

    !Get the command line arguments - needs to be handled using Fortrn
    argc = command_argument_count()
    do i = 1, argc
      call get_command_argument(i, temp)
      call op_set_args_c (argc, temp) !special function to set args
    end do


  end subroutine op_init_base_soa

  subroutine op_init_base ( diags, base_idx )

    ! formal parameter
    integer(4) :: diags
    integer(4) :: base_idx
    call op_init_base_soa(diags,base_idx,0)
  end subroutine op_init_base

  subroutine op_init(diags)
    integer(4) :: diags
    call op_init_base_soa(diags,1,0)
  end subroutine op_init
  
  subroutine op_init_soa(diags,soa)
    integer(4) :: diags
    integer(4) :: soa
    call op_init_base_soa(diags,1,soa)
  end subroutine op_init_soa

  subroutine op_mpi_init ( diags, global, local )

    ! formal parameter
    integer(4) :: diags
    integer(4) :: global
    integer(4) :: local

    ! local variables
    integer(c_int) :: argc = 0

    integer(4) :: rank2, ierr
    integer :: i
    character(kind=c_char,len=64)           :: temp

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
    call set_maps_base_c(1)

    call op_mpi_init_c ( argc, C_NULL_PTR, diags, global, local )

    !Get the command line arguments - needs to be handled using Fortrn
    argc = command_argument_count()
    do i = 1, argc
      call get_command_argument(i, temp)
      call op_set_args_c (argc, temp) !special function to set args
    end do

  end subroutine op_mpi_init


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

  subroutine op_decl_dat_real_8 ( set, datdim, type, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname
    character(kind=c_char,len=*) :: type

    if ( present ( opname ) ) then
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type//C_NULL_CHAR , 8, c_loc ( dat ), opName//C_NULL_CHAR )
    else
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type//C_NULL_CHAR , 8, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( data%dataCPtr, data%dataPtr )

    ! debugging

  end subroutine op_decl_dat_real_8

  subroutine op_decl_dat_real_8_2 ( set, datdim, type, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opName
    character(kind=c_char,len=*) :: type

    call op_decl_dat_real_8 ( set, datdim, type, dat, data, opname )

  end subroutine op_decl_dat_real_8_2

  subroutine op_decl_dat_real_8_3 ( set, datdim, type, dat, data, opname )

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    real(8), dimension(:,:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opName
    character(kind=c_char,len=*) :: type

    call op_decl_dat_real_8 ( set, datdim, type, dat, data, opname )

  end subroutine op_decl_dat_real_8_3

  subroutine op_decl_dat_integer_4 ( set, datdim, type, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(*), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname
    character(kind=c_char,len=*) :: type

    if ( present ( opname ) ) then
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type//C_NULL_CHAR, 4, c_loc ( dat ), opName//C_NULL_CHAR )
    else
      data%dataCPtr = op_decl_dat_c ( set%setCPtr, datdim, type//C_NULL_CHAR, 4, c_loc ( dat ), C_CHAR_'NONAME'//C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( data%dataCPtr, data%dataPtr )

  end subroutine op_decl_dat_integer_4

  subroutine op_decl_dat_integer_4_2 ( set, datdim, type, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname
    character(kind=c_char,len=*) :: type

    call op_decl_dat_integer_4 ( set, datdim, type, dat, data, opname )

  end subroutine op_decl_dat_integer_4_2

  subroutine op_decl_dat_integer_4_3 ( set, datdim, type, dat, data, opname )
    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    integer(4), dimension(:,:,:), intent(in), target :: dat
    type(op_dat) :: data
    character(kind=c_char,len=*), optional :: opname
    character(kind=c_char,len=*) :: type

    call op_decl_dat_integer_4 ( set, datdim, type, dat, data, opname )

  end subroutine op_decl_dat_integer_4_3

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   declarations of constants  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! All of these are no-ops in the reference implementation

  subroutine op_decl_const_integer_4 ( dat, constdim, opname )

    integer(4), dimension(:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "integer(4)" //C_NULL_CHAR, &
      & 4, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_integer_4

  subroutine op_decl_const_integer_2_4 ( dat, constdim, opname )

    integer(4), dimension(:,:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "integer(4)" //C_NULL_CHAR, &
      & 4, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_integer_2_4

  subroutine op_decl_const_real_2_8 ( dat, constdim, opname )

    real(8), dimension(:,:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "real(8)" //C_NULL_CHAR, &
      & 8, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_real_2_8

  subroutine op_decl_const_real_8 ( dat, constdim, opname )

    real(8), dimension(:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "real(8)" //C_NULL_CHAR, &
      & 8, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_real_8

  subroutine op_decl_const_scalar_integer_4 ( dat, constdim, opname )

    integer(4), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "integer(4)" //C_NULL_CHAR, &
      & 4, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_scalar_integer_4

  subroutine op_decl_const_scalar_real_8 ( dat, constdim, opname )

    real(8), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "real(8)" //C_NULL_CHAR, &
      & 8, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_scalar_real_8

  subroutine op_decl_const_logical ( dat, constdim, opname )

    logical, intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "logical" //C_NULL_CHAR, &
      & 4, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_logical

  subroutine op_decl_const_logical_1d ( dat, constdim, opname )

    logical, dimension(:), intent(in), target :: dat
    integer(kind=c_int), value :: constdim
    character(kind=c_char,len=*), optional :: opname

    call op_decl_const_char_c( constdim, "logical" //C_NULL_CHAR, &
      & 4, c_loc(dat), opname//C_NULL_CHAR);

  end subroutine op_decl_const_logical_1d

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
!#ifdef OP2_WITH_CUDAFOR
!    if (dat%dataCPtr .eq. C_NULL_PTR) then
!#else
    if ( isCNullPointer_c (dat%dataCPtr) .eqv. .true. ) then
!#endif
!      op_arg_dat_python = op_arg_dat_null_c (C_NULL_PTR, idx-1, C_NULL_PTR, -1, C_NULL_PTR, access-1)
      print *, "Error, NULL pointer for op_dat"
      op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
    else
      if (dat%dataPtr%dim .ne. dim) then
        print *, "Wrong dim",dim,dat%dataPtr%dim
        stop 1
      endif
      ! warning: access and idx are in FORTRAN style, while the C style is required here
      if ( map%mapPtr%dim .eq. 0 ) then
        ! OP_ID case (does not decrement idx)
        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
!        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx, C_NULL_PTR,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
      else
        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx-1, map%mapCPtr,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
!        op_arg_dat_python = op_arg_dat_c ( dat%dataCPtr, idx-1, map%mapCPtr,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
      endif
    endif

  end function op_arg_dat_python

  type(op_arg) function op_opt_arg_dat_python (opt, dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical :: opt
    type(op_dat) :: dat
    integer(kind=c_int) :: idx
    type(op_map) :: map
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access

    integer(kind=c_int) :: opt_int
    if (opt) then
        opt_int = 1
    else
        opt_int = 0
    endif

    ! warning: access and idx are in FORTRAN style, while the C style is required here
    if (opt) then
      if ( map%mapPtr%dim .eq. 0 ) then
        ! OP_ID case (does not decrement idx)
        op_opt_arg_dat_python = op_opt_arg_dat_c ( opt_int, dat%dataCPtr, idx, C_NULL_PTR,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
      else
        op_opt_arg_dat_python = op_opt_arg_dat_c ( opt_int, dat%dataCPtr, idx-1, map%mapCPtr,  dat%dataPtr%dim, type//C_NULL_CHAR, access-1 )
      endif
    else
      if ( map%mapPtr%dim .eq. 0 ) then
        ! OP_ID case (does not decrement idx)
        op_opt_arg_dat_python = op_opt_arg_dat_c ( opt_int, C_NULL_PTR, idx, C_NULL_PTR,  dim, type//C_NULL_CHAR, access-1 )
      else
        op_opt_arg_dat_python = op_opt_arg_dat_c ( opt_int, C_NULL_PTR, idx-1, map%mapCPtr,  dim, type//C_NULL_CHAR, access-1 )
      endif
!      op_opt_arg_dat_python = op_opt_arg_dat_c ( opt_int, C_NULL_PTR, idx, C_NULL_PTR,  dim, C_NULL_PTR, access-1 )
    endif

  end function op_opt_arg_dat_python

  INTEGER function op_get_size (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size = op_get_size_c ( set%setCPtr )

  end function op_get_size

  type(op_arg) function op_arg_gbl_python_r8_scalar ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_r8_scalar = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'double'//C_NULL_CHAR, 8, access-1 )
    !op_arg_gbl_python_r8_scalar = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_r8_scalar

  type(op_arg) function op_arg_gbl_python_i4_scalar ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_i4_scalar = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'int'//C_NULL_CHAR, 4, access-1 )
    !op_arg_gbl_python_i4_scalar = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_i4_scalar

  type(op_arg) function op_arg_gbl_python_logical_scalar ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical, target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_logical_scalar = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'bool'//C_NULL_CHAR, 1, access-1 )
    !op_arg_gbl_python_logical_scalar = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_logical_scalar

  type(op_arg) function op_arg_gbl_python_r8_1dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_r8_1dim = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'double'//C_NULL_CHAR, 8, access-1 )
    !op_arg_gbl_python_r8_1dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_r8_1dim

  type(op_arg) function op_arg_gbl_python_i4_1dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_i4_1dim = op_arg_gbl_c ( c_loc (dat), dim, C_CHAR_'int'//C_NULL_CHAR, 4, access-1 )
    !op_arg_gbl_python_i4_1dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_i4_1dim

  type(op_arg) function op_arg_gbl_python_logical_1dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical, dimension(*), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_logical_1dim = op_arg_gbl_c ( c_loc (dat(1)), dim, C_CHAR_'bool'//C_NULL_CHAR, 1, access-1 )
    !op_arg_gbl_python_logical_1dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_logical_1dim

#ifdef __GFORTRAN__
  function real_ptr ( arg )
    real(8), dimension(:,:), target :: arg
    real(8), target :: real_ptr

    real_ptr = arg(1, 1)
  end function
#else
#define real_ptr(arg) arg
#endif

  type(op_arg) function op_arg_gbl_python_r8_2dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), dimension(:,:), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_r8_2dim = op_arg_gbl_c ( c_loc (real_ptr(dat)), dim, C_CHAR_'double'//C_NULL_CHAR, 8, access-1 )
    !op_arg_gbl_python_r8_2dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_r8_2dim

#ifdef __GFORTRAN__
  function real_ptr3 ( arg )
    real(8), dimension(:,:,:), target, intent(in) :: arg
    real(8), target :: real_ptr3

    real_ptr3 = arg(1, 1, 1)
  end function
#else
#define real_ptr3(arg) arg
#endif

  type(op_arg) function op_arg_gbl_python_r8_3dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), dimension(:,:,:), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_r8_3dim = op_arg_gbl_c ( c_loc (dat(1,1,1)), dim, C_CHAR_'double'//C_NULL_CHAR, 8, access-1 )
    !op_arg_gbl_python_r8_3dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_r8_3dim

#ifdef __GFORTRAN__
  function int_ptr ( arg )
    integer(4), dimension(:,:), target :: arg
    integer(4), target :: int_ptr

    int_ptr = arg(1, 1)
  end function
#else
#define int_ptr(arg) arg
#endif

  type(op_arg) function op_arg_gbl_python_i4_2dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), dimension(:,:), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_i4_2dim = op_arg_gbl_c ( c_loc (int_ptr(dat)), dim, C_CHAR_'int'//C_NULL_CHAR, 4, access-1 )
    !op_arg_gbl_python_i4_2dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_i4_2dim

  type(op_arg) function op_arg_gbl_python_logical_2dim ( dat, dim, type, access )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    logical, dimension(:,:), target :: dat
    integer(kind=c_int) :: dim
    integer(kind=c_int) :: access
    character(kind=c_char,len=*) :: type

    ! warning: access is in FORTRAN style, while the C style is required here
    op_arg_gbl_python_logical_2dim = op_arg_gbl_c ( c_loc (dat(1, 1)), dim, C_CHAR_'bool'//C_NULL_CHAR, 1, access-1 )
    !op_arg_gbl_python_logical_2dim = op_arg_gbl_c ( dat%dataCPtr, dat%dataPtr%dim, dat%dataPtr%type, access-1 )

  end function op_arg_gbl_python_logical_2dim

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

  subroutine op_timers_core ( et )

    real(kind=c_double) :: et

    real(kind=c_double) :: cpu = 0

    call op_timers_core_f ( cpu, et )

  end subroutine op_timers_core

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

  subroutine op_print_dat_to_txtfile (dat, fileName)

    type(op_dat) :: dat
    character(len=*) :: fileName

    call op_print_dat_to_txtfile_c (dat%dataPtr, fileName)

  end subroutine op_print_dat_to_txtfile

  subroutine op_mpi_rank (rank)

    integer(kind=c_int) :: rank

    call op_mpi_rank_c (rank)

  end subroutine op_mpi_rank

  subroutine op_print (line)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    character(kind=c_char,len=*) :: line

    call op_print_c (line//C_NULL_CHAR)

  end subroutine

  subroutine op_fetch_data_real_8 ( dat, data )

    real(8), dimension(*), target :: data
    type(op_dat) :: dat

    call op_fetch_data_c ( dat%dataPtr, c_loc (data))

  end subroutine op_fetch_data_real_8

  subroutine op_fetch_data_real_4 ( dat, data )

    real, dimension(*), target :: data
    type(op_dat) :: dat

    call op_fetch_data_c ( dat%dataPtr, c_loc (data))

  end subroutine op_fetch_data_real_4

  subroutine op_fetch_data_integer_4 ( dat, data )

    integer(4), dimension(*), target :: data
    type(op_dat) :: dat

    call op_fetch_data_c ( dat%dataPtr, c_loc (data))

  end subroutine op_fetch_data_integer_4

  subroutine op_fetch_data_idx_real_8 ( dat, data, low, high )

    real(8), dimension(*), target :: data
    type(op_dat) :: dat
    integer(kind=c_int), value :: high
    integer(kind=c_int), value :: low

    call op_fetch_data_idx_c ( dat%dataPtr, c_loc (data), low-1, high-1)

  end subroutine op_fetch_data_idx_real_8

  subroutine op_fetch_data_idx_real_4 ( dat, data, low, high )

    real, dimension(*), target :: data
    type(op_dat) :: dat
    integer(kind=c_int), value :: high
    integer(kind=c_int), value :: low

    call op_fetch_data_idx_c ( dat%dataPtr, c_loc (data), low-1, high-1)

  end subroutine op_fetch_data_idx_real_4

  subroutine op_fetch_data_idx_integer_4 ( dat, data, low, high )

    integer(4), dimension(*), target :: data
    type(op_dat) :: dat
    integer(kind=c_int), value :: high
    integer(kind=c_int), value :: low

    call op_fetch_data_idx_c ( dat%dataPtr, c_loc (data), low-1, high-1)

  end subroutine op_fetch_data_idx_integer_4

  subroutine op_import_init_size ( nprocs, proclist_ptr, mark, handle )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer                       :: nprocs
    integer, dimension(:), target :: proclist_ptr
    type(op_dat)                  :: mark
    type(op_import_handle)        :: handle

    handle%importCptr = op_import_init_size_c ( nprocs, c_loc(proclist_ptr), mark%dataPtr)

    call c_f_pointer ( handle%importCPtr, handle%importPtr )

  end subroutine op_import_init_size

  subroutine op_import_init ( exp_handle, coords, mark, handle )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_export_handle) :: exp_handle
    type(op_dat)           :: coords
    type(op_dat)           :: mark
    type(op_import_handle) :: handle

    handle%importCptr = op_import_init_c ( exp_handle%exportPtr, coords%dataPtr, mark%dataPtr)

    call c_f_pointer ( handle%importCPtr, handle%importPtr )

  end subroutine op_import_init

  subroutine op_export_init ( nprocs, proclist_ptr, cells2Nodes, sp_nodes, coords, mark, handle )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer                       :: nprocs
    integer, dimension(:), target :: proclist_ptr
    type(op_map)                  :: cells2Nodes
    type(op_set)                  :: sp_nodes
    type(op_dat)                  :: coords
    type(op_dat)                  :: mark
    type(op_export_handle)        :: handle

    handle%exportCptr = op_export_init_c ( nprocs, c_loc(proclist_ptr), cells2Nodes%mapPtr, sp_nodes%setPtr, coords%dataPtr, mark%dataPtr)

    call c_f_pointer ( handle%exportCPtr, handle%exportPtr )

  end subroutine op_export_init

  subroutine op_export_data ( handle, dat )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_export_handle)      :: handle
    type(op_dat)                :: dat

    ! local variables

    call op_export_data_c ( handle%exportPtr, dat%dataPtr )

  end subroutine op_export_data


  subroutine op_import_data ( handle, dat )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_import_handle)      :: handle
    type(op_dat)                :: dat

    ! local variables

    call op_import_data_c ( handle%importPtr, dat%dataPtr )

  end subroutine op_import_data


  subroutine op_inc_theta ( handle, bc_id, dtheta_exp, dtheta_imp )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_export_handle)             :: handle
    integer, dimension(:), target      :: bc_id
    real(kind=8), dimension(:), target :: dtheta_exp
    real(kind=8), dimension(:), target :: dtheta_imp

    ! local variables

    call op_inc_theta_c ( handle%exportPtr, c_loc(bc_id), c_loc(dtheta_exp), c_loc(dtheta_imp) )

  end subroutine op_inc_theta

  subroutine op_theta_init ( handle, bc_id, dtheta_exp, dtheta_imp, alpha )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_export_handle)             :: handle
    integer, dimension(:), target      :: bc_id
    real(kind=8), dimension(:), target :: dtheta_exp
    real(kind=8), dimension(:), target :: dtheta_imp
    real(kind=8), dimension(:), target :: alpha

    ! local variables

    call op_theta_init_c ( handle%exportPtr, c_loc(bc_id), c_loc(dtheta_exp), c_loc(dtheta_imp), c_loc(alpha) )

  end subroutine op_theta_init

end module OP2_Fortran_Declarations

