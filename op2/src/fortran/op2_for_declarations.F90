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

#define UNUSED(x) if (.false.) print *, loc(x)
#define OP2_ARG_POINTERS

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
  integer(c_int) :: OP_WORK = 7

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
#ifdef OP2_ARG_POINTERS
  integer(4) :: OP_ID(2)
#else
  type(op_map) :: OP_ID
#endif
  type(op_map) :: OP_GBL

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

    subroutine op_disable_device_execution_c(disable) bind(C,name='op_disable_device_execution')

      use, intrinsic :: ISO_C_BINDING
      logical(kind=c_bool), value :: disable

    end subroutine op_disable_device_execution_c

    logical(kind=c_bool) function op_check_whitelist_c(name) bind(C,name='op_check_whitelist')

      use, intrinsic :: ISO_C_BINDING
      character(kind=c_char) :: name(*)

    end function op_check_whitelist_c

    subroutine op_disable_mpi_reductions_c(disable) bind(C,name='op_disable_mpi_reductions')

      use, intrinsic :: ISO_C_BINDING
      logical(kind=c_bool), value :: disable

    end subroutine op_disable_mpi_reductions_c

    subroutine op_register_set_c (idx, set) BIND(C,name='op_register_set')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: idx
      type(c_ptr), value, intent(in)           :: set
    end subroutine op_register_set_c

    type(c_ptr) function op_get_set_c ( idx ) BIND(C,name='op_get_set')
      use, intrinsic :: ISO_C_BINDING
      import :: op_set_core
      integer(kind=c_int), intent(in), value :: idx
    end function op_get_set_c

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

    INTEGER(kind=c_int) function op_get_global_set_offset_c ( set ) BIND(C,name='op_get_global_set_offset')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function

    INTEGER(kind=c_int) function op_get_size_local_core_c ( set ) BIND(C,name='op_get_size_local_core')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function

    INTEGER(kind=c_int) function op_get_size_local_c ( set ) BIND(C,name='op_get_size_local')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function

    INTEGER(kind=c_int) function op_get_size_local_exec_c ( set ) BIND(C,name='op_get_size_local_exec')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function

    INTEGER(kind=c_int) function op_get_size_local_full_c ( set ) BIND(C,name='op_get_size_local_full')
      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function

    integer(8) function op_get_g_index_c(set) bind(C, name='op_get_g_index')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set
      type(c_ptr), value, intent(in) :: set

    end function op_get_g_index_c

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

    type(c_ptr) function op_decl_dat_overlay_ptr_c(set, dat) BIND(C,name='op_decl_dat_overlay_ptr')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in) :: set
      type(c_ptr), value, intent(in) :: dat

    end function op_decl_dat_overlay_ptr_c

    type(c_ptr) function op_decl_dat_temp_char_c ( set, datdim, type, datsize, name ) BIND(C,name='op_decl_dat_temp_char')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_dat_core

      type(c_ptr), value, intent(in)           :: set
      integer(kind=c_int), value               :: datdim, datsize
      character(kind=c_char,len=1), intent(in) :: type(*)
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function op_decl_dat_temp_char_c

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

    function op_arg_dat_ptr_c ( opt, dat, idx, map, dim, type, acc ) BIND(C,name='op_arg_dat_ptr')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_dat_ptr_c

      integer(kind=c_int), value :: opt
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1) :: type(*)
      integer(kind=c_int), value :: acc

    end function op_arg_dat_ptr_c

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

    function op_arg_gbl_ptr_c ( opt, dat, dim, type, size, acc ) BIND(C,name='op_opt_arg_gbl_copy')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_gbl_ptr_c

      integer(kind=c_int), value :: opt
      type(c_ptr), value :: dat
      integer(kind=c_int), value :: dim
      character(kind=c_char), dimension(*) :: type
      integer(kind=c_int), value :: size
      integer(kind=c_int), value :: acc

    end function op_arg_gbl_ptr_c

    function op_arg_info_c ( dat, dim, type, size, ref ) BIND(C,name='op_arg_info_copy')

      use, intrinsic :: ISO_C_BINDING

      import :: op_arg

      type(op_arg) :: op_arg_info_c

      type(c_ptr), value :: dat
      integer(kind=c_int), value :: dim
      character(kind=c_char), dimension(*) :: type
      integer(kind=c_int), value :: size
      integer(kind=c_int), value :: ref

    end function op_arg_info_c

    function op_arg_idx_c(idx, map) BIND(C,name='op_arg_idx')
      use, intrinsic :: ISO_C_BINDING

      import :: op_arg
      type(op_arg) :: op_arg_idx_c
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
    end function op_arg_idx_c

    function op_arg_idx_ptr_c(idx, map) BIND(C,name='op_arg_idx_ptr')
      use, intrinsic :: ISO_C_BINDING

      import :: op_arg
      type(op_arg) :: op_arg_idx_ptr_c
      integer(kind=c_int), value :: idx
      type(c_ptr), value, intent(in) :: map
    end function op_arg_idx_ptr_c

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

    subroutine op_memalloc ( ptr, bytes ) BIND(C,name='op_malloc2')
      use, intrinsic :: ISO_C_BINDING

      integer*8 :: ptr
      integer(kind=c_int) :: bytes

    end subroutine op_memalloc

    function op_free_dat_temp_c ( dat ) BIND(C,name='op_free_dat_temp_char')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in) :: dat

      integer(kind=c_int) op_free_dat_temp_c

    end function

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

   type(c_ptr) function op_mpi_get_data_c(dat) BIND(C, name='op_mpi_get_data')

     use, intrinsic :: ISO_C_BINDING
     type(c_ptr), value :: dat

   end function op_mpi_get_data_c

   subroutine op_mpi_free_data_c(dat) BIND(C, name='op_mpi_free_data')

     use, intrinsic :: ISO_C_BINDING
     type(c_ptr), value :: dat

   end subroutine op_mpi_free_data_c

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

    subroutine op_print_dat_to_txtfile_c (dat, fileName) BIND(C,name='op_print_dat_to_txtfile')
      use, intrinsic :: ISO_C_BINDING

      import :: op_dat_core

      type(op_dat_core) :: dat
      character(len=1,kind=c_char) :: fileName(*)

    end subroutine op_print_dat_to_txtfile_c

    subroutine op_print_dat_to_txtfile2_c (dat, fileName) BIND(C,name='op_print_dat_to_txtfile2')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value ::    dat
      character(len=1,kind=c_char) :: fileName(*)

    end subroutine op_print_dat_to_txtfile2_c

    logical(kind=c_bool) function isCNullPointer_c (ptr) BIND(C,name='isCNullPointer')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value :: ptr
    end function isCNullPointer_c

    subroutine op_timing_output () BIND(C,name='op_timing_output')
    end subroutine op_timing_output

    subroutine op_timing2_start_c(name) BIND(C,name='op_timing2_start')
      use ISO_C_BINDING
      character(kind=c_char) :: name(*)
    end subroutine op_timing2_start_c

    subroutine op_timing2_enter_c(name) BIND(C,name='op_timing2_enter')
      use ISO_C_BINDING
      character(kind=c_char) :: name(*)
    end subroutine op_timing2_enter_c

    subroutine op_timing2_enter_kernel_c(name, target, variant) BIND(C,name='op_timing2_enter_kernel')
      use ISO_C_BINDING
      character(kind=c_char) :: name(*), target(*), variant(*)
    end subroutine op_timing2_enter_kernel_c

    subroutine op_timing2_next_c(name) BIND(C,name='op_timing2_next')
      use ISO_C_BINDING
      character(kind=c_char) :: name(*)
    end subroutine op_timing2_next_c

    subroutine op_timing2_exit() BIND(C,name='op_timing2_exit')
    end subroutine op_timing2_exit

    subroutine op_timing2_finish() BIND(C,name='op_timing2_finish')
    end subroutine op_timing2_finish

    subroutine op_timing2_output() BIND(C,name='op_timing2_output')
    end subroutine op_timing2_output

    subroutine op_timing2_output_json_c(filename) BIND(C,name='op_timing2_output_json')
      use ISO_C_BINDING
      character(kind=c_char) :: filename(*)
    end subroutine op_timing2_output_json_c

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

   INTEGER(8) function op_get_data_ptr_c ( data ) BIND(C,name='op_get_data_ptr')
     use, intrinsic :: ISO_C_BINDING
     import :: op_dat_core
     type(op_dat_core) :: data
   end function

   INTEGER(8) function op_get_data_ptr_int_c ( data ) BIND(C,name='op_get_data_ptr2')
     use, intrinsic :: ISO_C_BINDING
     integer(8), value :: data
   end function

   INTEGER(8) function op_reset_data_ptr_c ( data, mode ) BIND(C,name='op_reset_data_ptr')
     use, intrinsic :: ISO_C_BINDING
     type(c_ptr), value, intent(in) :: data
     integer(kind=c_int), value :: mode
   end function

   INTEGER(8) function op_get_map_ptr_c ( map ) BIND(C,name='op_get_map_ptr')
     use, intrinsic :: ISO_C_BINDING
     import :: op_map_core
     type(op_map_core) :: map
   end function

   INTEGER(8) function op_reset_map_ptr_c ( map ) BIND(C,name='op_reset_map_ptr')
     use, intrinsic :: ISO_C_BINDING
     type(c_ptr), value, intent(in) :: map
   end function

   INTEGER(8) function op_copy_map_to_fort_c ( map ) BIND(C,name='op_copy_map_to_fort')
     use, intrinsic :: ISO_C_BINDING
     type(c_ptr), value, intent(in) :: map
   end function


   subroutine op_check_fortran_type_int_c(type) bind(C, name='op_check_fortran_type_int')

     use, intrinsic :: ISO_C_BINDING
     character(kind=c_char,len=1) :: type(*)

   end subroutine op_check_fortran_type_int_c

   subroutine op_check_fortran_type_float_c(type) bind(C, name='op_check_fortran_type_float')

     use, intrinsic :: ISO_C_BINDING
     character(kind=c_char,len=1) :: type(*)

   end subroutine op_check_fortran_type_float_c

   subroutine op_check_fortran_type_double_c(type) bind(C, name='op_check_fortran_type_double')

     use, intrinsic :: ISO_C_BINDING
     character(kind=c_char,len=1) :: type(*)

   end subroutine op_check_fortran_type_double_c

   subroutine op_check_fortran_type_bool_c(type) bind(C, name='op_check_fortran_type_bool')

     use, intrinsic :: ISO_C_BINDING
     character(kind=c_char,len=1) :: type(*)

   end subroutine op_check_fortran_type_bool_c


  end interface

#define DIM_0 0
#define DIM_1 1
#define DIM_2 2
#define DIM_3 3
#define DIM_4 4
#define DIM_5 5

#define MAP_DIM_1 1
#define MAP_DIM_2 2

#define DECL_DIM_0
#define DECL_DIM_1 , dimension(*)
#define DECL_DIM_2 , dimension(:, :)
#define DECL_DIM_3 , dimension(:, :, :)
#define DECL_DIM_4 , dimension(:, :, :, :)
#define DECL_DIM_5 , dimension(:, :, :, :, :)

#define TYPE_INTEGER_4 integer(4)
#define TYPE_INTEGER_8 integer(8)
#define TYPE_REAL_4    real(4)
#define TYPE_REAL_8    real(8)
#define TYPE_LOGICAL   logical
#define TYPE_STRING    character(kind=c_char, len=*)

#define TYPE_SIZE_INTEGER_4 4
#define TYPE_SIZE_INTEGER_8 8
#define TYPE_SIZE_REAL_4    4
#define TYPE_SIZE_REAL_8    8
#define TYPE_SIZE_LOGICAL   1

#define TYPE_STR_INTEGER_4 C_CHAR_"int" /@/ C_NULL_CHAR
#define TYPE_STR_REAL_4    C_CHAR_"float" /@/ C_NULL_CHAR
#define TYPE_STR_REAL_8    C_CHAR_"double" /@/ C_NULL_CHAR
#define TYPE_STR_LOGICAL   C_CHAR_"bool" /@/ C_NULL_CHAR

#define TYPE_CHECK_INTEGER_4(type) op_check_fortran_type_int_c(type)
#define TYPE_CHECK_REAL_4(type)    op_check_fortran_type_float_c(type)
#define TYPE_CHECK_REAL_8(type)    op_check_fortran_type_double_c(type)
#define TYPE_CHECK_LOGICAL(type)   op_check_fortran_type_bool_c(type)


! ---- op_decl_dat wrappers ---- !

#define INTF_DECL_DAT(TYPE, DIM) INTF_DECL_DAT_(TYPE, DIM)
#define INTF_DECL_DAT_(TYPE, DIM) op_decl_dat_##TYPE##_##DIM

  interface op_decl_dat
    module procedure INTF_DECL_DAT(INTEGER_4, DIM_1), &
                     INTF_DECL_DAT(INTEGER_4, DIM_2), &
                     INTF_DECL_DAT(INTEGER_4, DIM_3), &
                     INTF_DECL_DAT(REAL_4,    DIM_1), &
                     INTF_DECL_DAT(REAL_4,    DIM_2), &
                     INTF_DECL_DAT(REAL_4,    DIM_3), &
                     INTF_DECL_DAT(REAL_8,    DIM_1), &
                     INTF_DECL_DAT(REAL_8,    DIM_2), &
                     INTF_DECL_DAT(REAL_8,    DIM_3)
  end interface op_decl_dat

#define DECL_DECL_DAT(TYPE, DIM) DECL_DECL_DAT_(TYPE, DIM)
#define DECL_DECL_DAT_(TYPE, DIM)                                                                                      \
subroutine INTF_DECL_DAT(TYPE, DIM) (set, dim, type, data, dat, name)                                                 @\
                                                                                                                      @\
    type(op_set) :: set                                                                                               @\
    integer(kind=c_int) :: dim                                                                                        @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
                                                                                                                      @\
    type(op_dat) :: dat                                                                                               @\
                                                                                                                      @\
    character(kind=c_char, len=*), optional :: name                                                                   @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    character(kind=c_char, len=:), allocatable :: name2                                                               @\
                                                                                                                      @\
    if (present(name)) then                                                                                           @\
      name2 = name /@/ C_NULL_CHAR                                                                                    @\
    else                                                                                                              @\
      name2 = C_CHAR_"unnamed" /@/ C_NULL_CHAR                                                                        @\
    end if                                                                                                            @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    dat%dataCPtr = op_decl_dat_c(set%setCPtr, dim, TYPE_STR_##TYPE, TYPE_SIZE_##TYPE, c_loc(data), name2)             @\
    call c_f_pointer(dat%dataCPtr, dat%dataPtr)                                                                       @\
                                                                                                                      @\
end subroutine INTF_DECL_DAT(TYPE, DIM)


! ---- op_decl_dat_overlay wrappers ---- !

#define INTF_DECL_DAT_OVERLAY(TYPE, DIM) INTF_DECL_DAT_OVERLAY_(TYPE, DIM)
#define INTF_DECL_DAT_OVERLAY_(TYPE, DIM) op_decl_dat_overlay_##TYPE##_##DIM

  interface op_decl_dat_overlay
    module procedure INTF_DECL_DAT_OVERLAY(INTEGER_4, DIM_1), &
                     INTF_DECL_DAT_OVERLAY(INTEGER_4, DIM_2), &
                     INTF_DECL_DAT_OVERLAY(INTEGER_4, DIM_3), &
                     INTF_DECL_DAT_OVERLAY(REAL_4,    DIM_1), &
                     INTF_DECL_DAT_OVERLAY(REAL_4,    DIM_2), &
                     INTF_DECL_DAT_OVERLAY(REAL_4,    DIM_3), &
                     INTF_DECL_DAT_OVERLAY(REAL_4,    DIM_4), &
                     INTF_DECL_DAT_OVERLAY(REAL_4,    DIM_5), &
                     INTF_DECL_DAT_OVERLAY(REAL_8,    DIM_1), &
                     INTF_DECL_DAT_OVERLAY(REAL_8,    DIM_2), &
                     INTF_DECL_DAT_OVERLAY(REAL_8,    DIM_3), &
                     INTF_DECL_DAT_OVERLAY(REAL_8,    DIM_4), &
                     INTF_DECL_DAT_OVERLAY(REAL_8,    DIM_5)
  end interface op_decl_dat_overlay

#define DECL_DECL_DAT_OVERLAY(TYPE, DIM) DECL_DECL_DAT_OVERLAY_(TYPE, DIM)
#define DECL_DECL_DAT_OVERLAY_(TYPE, DIM)                                                                              \
integer(8) function INTF_DECL_DAT_OVERLAY(TYPE, DIM) (set, data)                                                      @\
                                                                                                                      @\
    type(op_set) :: set                                                                                               @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
                                                                                                                      @\
    type(op_dat) :: overlay_dat                                                                                       @\
                                                                                                                      @\
    overlay_dat%dataCPtr = op_decl_dat_overlay_ptr_c(set%setCPtr, c_loc(data))                                        @\
    call c_f_pointer(overlay_dat%dataCPtr, overlay_dat%dataPtr)                                                       @\
                                                                                                                      @\
    INTF_DECL_DAT_OVERLAY(TYPE, DIM) = overlay_dat%dataPtr%index                                                      @\
                                                                                                                      @\
end function INTF_DECL_DAT_OVERLAY(TYPE, DIM)


! ---- op_decl_const wrappers ---- !

#define INTF_DECL_CONST(TYPE, DIM) INTF_DECL_CONST_(TYPE, DIM)
#define INTF_DECL_CONST_(TYPE, DIM) op_decl_const_##TYPE##_##DIM

  interface op_decl_const
    module procedure INTF_DECL_CONST(INTEGER_4, DIM_0), &
                     INTF_DECL_CONST(INTEGER_4, DIM_1), &
                     INTF_DECL_CONST(INTEGER_4, DIM_2), &
                     INTF_DECL_CONST(INTEGER_8, DIM_0), &
                     INTF_DECL_CONST(INTEGER_8, DIM_1), &
                     INTF_DECL_CONST(INTEGER_8, DIM_2), &
                     INTF_DECL_CONST(REAL_4,    DIM_0), &
                     INTF_DECL_CONST(REAL_4,    DIM_1), &
                     INTF_DECL_CONST(REAL_4,    DIM_2), &
                     INTF_DECL_CONST(REAL_8,    DIM_0), &
                     INTF_DECL_CONST(REAL_8,    DIM_1), &
                     INTF_DECL_CONST(REAL_8,    DIM_2), &
                     INTF_DECL_CONST(LOGICAL,   DIM_0), &
                     INTF_DECL_CONST(STRING,    DIM_0)
  end interface op_decl_const

#define DECL_DECL_CONST(TYPE, DIM) DECL_DECL_CONST_(TYPE, DIM)
#define DECL_DECL_CONST_(TYPE, DIM)                                                                                    \
  subroutine INTF_DECL_CONST(TYPE, DIM) (data, dim, name)                                                             @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
    integer(kind=c_int) :: dim                                                                                        @\
    character(kind=c_char,len=*), optional :: name                                                                    @\
                                                                                                                      @\
    UNUSED(data)                                                                                                      @\
    UNUSED(dim)                                                                                                       @\
    UNUSED(name)                                                                                                      @\
                                                                                                                      @\
  end subroutine INTF_DECL_CONST(TYPE, DIM)


! ---- op_arg_dat wrappers ---- !

#define INTF_ARG_DAT(TYPE, DIM, MAP_DIM) INTF_ARG_DAT_(TYPE, DIM, MAP_DIM)
#define INTF_ARG_DAT_(TYPE, DIM, MAP_DIM) op_arg_dat_##TYPE##_##DIM##_##MAP_DIM

  interface op_arg_dat
    module procedure op_arg_dat_python, &
                     op_arg_dat_python_OP_ID, &
                     INTF_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_1), &
                     INTF_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_1), &
                     INTF_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_1), &
                     INTF_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_2), &
                     INTF_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_2), &
                     INTF_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_1), &
                     INTF_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_2), &
                     INTF_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_2)
  end interface op_arg_dat

#define DECL_ARG_DAT(TYPE, DIM, MAP_DIM) DECL_ARG_DAT_(TYPE, DIM, MAP_DIM)
#define DECL_ARG_DAT_(TYPE, DIM, MAP_DIM)                                                                              \
  type(op_arg) function INTF_ARG_DAT(TYPE, DIM, MAP_DIM) (data, idx, map, dim, type, access)                          @\
                                                                                                                      @\
    use, intrinsic :: ISO_C_BINDING                                                                                   @\
    implicit none                                                                                                     @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
    integer(4) DECL_DIM_##MAP_DIM, target :: map                                                                      @\
                                                                                                                      @\
    integer(kind=c_int) :: idx, dim, access                                                                           @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    integer(kind=c_int) :: opt = 1                                                                                    @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    INTF_ARG_DAT(TYPE, DIM, MAP_DIM) = op_arg_dat_ptr_c(opt, c_loc(data), idx - 1, c_loc(map), dim, &                 @\
                                                        TYPE_STR_##TYPE, access - 1)                                  @\
                                                                                                                      @\
  end function INTF_ARG_DAT(TYPE, DIM, MAP_DIM)


! ---- op_opt_arg_dat wrappers ---- !

#define INTF_OPT_ARG_DAT(TYPE, DIM, MAP_DIM) INTF_OPT_ARG_DAT_(TYPE, DIM, MAP_DIM)
#define INTF_OPT_ARG_DAT_(TYPE, DIM, MAP_DIM) op_opt_arg_dat_##TYPE##_##DIM##_##MAP_DIM

  interface op_opt_arg_dat
    module procedure op_opt_arg_dat_python, &
                     op_opt_arg_dat_python_OP_ID, &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_1), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_2), &
                     INTF_OPT_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_2)
  end interface op_opt_arg_dat

#define DECL_OPT_ARG_DAT(TYPE, DIM, MAP_DIM) DECL_OPT_ARG_DAT_(TYPE, DIM, MAP_DIM)
#define DECL_OPT_ARG_DAT_(TYPE, DIM, MAP_DIM)                                                                          \
  type(op_arg) function INTF_OPT_ARG_DAT(TYPE, DIM, MAP_DIM) (opt, data, idx, map, dim, type, access)                 @\
                                                                                                                      @\
    use, intrinsic :: ISO_C_BINDING                                                                                   @\
    implicit none                                                                                                     @\
                                                                                                                      @\
    logical :: opt                                                                                                    @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
    integer(4) DECL_DIM_##MAP_DIM, target :: map                                                                      @\
                                                                                                                      @\
    integer(kind=c_int) :: idx, dim, access                                                                           @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    integer(kind=c_int) :: c_opt                                                                                      @\
                                                                                                                      @\
    if (opt) then                                                                                                     @\
      c_opt = 1                                                                                                       @\
    else                                                                                                              @\
      c_opt = 0                                                                                                       @\
    end if                                                                                                            @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    INTF_OPT_ARG_DAT(TYPE, DIM, MAP_DIM) = op_arg_dat_ptr_c(c_opt, c_loc(data), idx - 1, c_loc(map), dim, &           @\
                                                            TYPE_STR_##TYPE, access - 1)                              @\
                                                                                                                      @\
  end function INTF_OPT_ARG_DAT(TYPE, DIM, MAP_DIM)


! ---- op_arg_gbl wrappers ---- !

#define INTF_ARG_GBL(TYPE, DIM) INTF_ARG_GBL_(TYPE, DIM)
#define INTF_ARG_GBL_(TYPE, DIM) op_arg_gbl_##TYPE##_##DIM

  interface op_arg_gbl
    module procedure INTF_ARG_GBL(INTEGER_4, DIM_0), &
                     INTF_ARG_GBL(INTEGER_4, DIM_1), &
                     INTF_ARG_GBL(INTEGER_4, DIM_2), &
                     INTF_ARG_GBL(REAL_4,    DIM_0), &
                     INTF_ARG_GBL(REAL_4,    DIM_1), &
                     INTF_ARG_GBL(REAL_4,    DIM_2), &
                     INTF_ARG_GBL(REAL_4,    DIM_3), &
                     INTF_ARG_GBL(REAL_8,    DIM_0), &
                     INTF_ARG_GBL(REAL_8,    DIM_1), &
                     INTF_ARG_GBL(REAL_8,    DIM_2), &
                     INTF_ARG_GBL(REAL_8,    DIM_3), &
                     INTF_ARG_GBL(LOGICAL,   DIM_0), &
                     INTF_ARG_GBL(LOGICAL,   DIM_1), &
                     INTF_ARG_GBL(LOGICAL,   DIM_2)
  end interface op_arg_gbl

#define DECL_ARG_GBL(TYPE, DIM) DECL_ARG_GBL_(TYPE, DIM)
#define DECL_ARG_GBL_(TYPE, DIM)                                                                                       \
  type(op_arg) function INTF_ARG_GBL(TYPE, DIM) (data, dim, type, access)                                             @\
                                                                                                                      @\
    use, intrinsic :: ISO_C_BINDING                                                                                   @\
    implicit none                                                                                                     @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
                                                                                                                      @\
    integer(kind=c_int) :: dim, access                                                                                @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    INTF_ARG_GBL(TYPE, DIM) = op_arg_gbl_c(c_loc(data), dim, TYPE_STR_##TYPE, TYPE_SIZE_##TYPE, access - 1)           @\
                                                                                                                      @\
  end function INTF_ARG_GBL(TYPE, DIM)


! ---- op_opt_arg_gbl wrappers ---- !

#define INTF_OPT_ARG_GBL(TYPE, DIM) INTF_OPT_ARG_GBL_(TYPE, DIM)
#define INTF_OPT_ARG_GBL_(TYPE, DIM) op_opt_arg_gbl_##TYPE##_##DIM

  interface op_opt_arg_gbl
    module procedure INTF_OPT_ARG_GBL(INTEGER_4, DIM_0), &
                     INTF_OPT_ARG_GBL(INTEGER_4, DIM_1), &
                     INTF_OPT_ARG_GBL(INTEGER_4, DIM_2), &
                     INTF_OPT_ARG_GBL(REAL_4,    DIM_0), &
                     INTF_OPT_ARG_GBL(REAL_4,    DIM_1), &
                     INTF_OPT_ARG_GBL(REAL_4,    DIM_2), &
                     INTF_OPT_ARG_GBL(REAL_4,    DIM_3), &
                     INTF_OPT_ARG_GBL(REAL_8,    DIM_0), &
                     INTF_OPT_ARG_GBL(REAL_8,    DIM_1), &
                     INTF_OPT_ARG_GBL(REAL_8,    DIM_2), &
                     INTF_OPT_ARG_GBL(REAL_8,    DIM_3), &
                     INTF_OPT_ARG_GBL(LOGICAL,   DIM_0), &
                     INTF_OPT_ARG_GBL(LOGICAL,   DIM_1), &
                     INTF_OPT_ARG_GBL(LOGICAL,   DIM_2)
  end interface op_opt_arg_gbl

#define DECL_OPT_ARG_GBL(TYPE, DIM) DECL_OPT_ARG_GBL_(TYPE, DIM)
#define DECL_OPT_ARG_GBL_(TYPE, DIM)                                                                                   \
  type(op_arg) function INTF_OPT_ARG_GBL(TYPE, DIM) (opt, data, dim, type, access)                                    @\
                                                                                                                      @\
    use, intrinsic :: ISO_C_BINDING                                                                                   @\
    implicit none                                                                                                     @\
                                                                                                                      @\
    logical :: opt                                                                                                    @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
                                                                                                                      @\
    integer(kind=c_int) :: dim, access                                                                                @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    integer(kind=c_int) :: c_opt                                                                                      @\
                                                                                                                      @\
    if (opt) then                                                                                                     @\
      c_opt = 1                                                                                                       @\
    else                                                                                                              @\
      c_opt = 0                                                                                                       @\
    end if                                                                                                            @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    INTF_OPT_ARG_GBL(TYPE, DIM) = op_arg_gbl_ptr_c(c_opt, c_loc(data), dim, &                                         @\
                                                   TYPE_STR_##TYPE, TYPE_SIZE_##TYPE, access - 1)                     @\
                                                                                                                      @\
  end function INTF_OPT_ARG_GBL(TYPE, DIM)


! ---- op_arg_info wrappers ---- !

#define INTF_ARG_INFO(TYPE, DIM) INTF_ARG_INFO_(TYPE, DIM)
#define INTF_ARG_INFO_(TYPE, DIM) op_arg_info_##TYPE##_##DIM

  interface op_arg_info
    module procedure INTF_ARG_INFO(INTEGER_4, DIM_0), &
                     INTF_ARG_INFO(INTEGER_4, DIM_1), &
                     INTF_ARG_INFO(INTEGER_4, DIM_2), &
                     INTF_ARG_INFO(REAL_4,    DIM_0), &
                     INTF_ARG_INFO(REAL_4,    DIM_1), &
                     INTF_ARG_INFO(REAL_4,    DIM_2), &
                     INTF_ARG_INFO(REAL_4,    DIM_3), &
                     INTF_ARG_INFO(REAL_8,    DIM_0), &
                     INTF_ARG_INFO(REAL_8,    DIM_1), &
                     INTF_ARG_INFO(REAL_8,    DIM_2), &
                     INTF_ARG_INFO(REAL_8,    DIM_3), &
                     INTF_ARG_INFO(LOGICAL,   DIM_0), &
                     INTF_ARG_INFO(LOGICAL,   DIM_1), &
                     INTF_ARG_INFO(LOGICAL,   DIM_2)
  end interface op_arg_info

#define DECL_ARG_INFO(TYPE, DIM) DECL_ARG_INFO_(TYPE, DIM)
#define DECL_ARG_INFO_(TYPE, DIM)                                                                                      \
  type(op_arg) function INTF_ARG_INFO(TYPE, DIM) (data, dim, type, ref)                                               @\
                                                                                                                      @\
    use, intrinsic :: ISO_C_BINDING                                                                                   @\
    implicit none                                                                                                     @\
                                                                                                                      @\
    TYPE_##TYPE DECL_DIM_##DIM, target :: data                                                                        @\
                                                                                                                      @\
    integer(kind=c_int) :: dim, ref                                                                                   @\
    character(kind=c_char, len=*) :: type                                                                             @\
                                                                                                                      @\
    call TYPE_CHECK_##TYPE(type)                                                                                      @\
                                                                                                                      @\
    INTF_ARG_INFO(TYPE, DIM) = op_arg_info_c(c_loc(data), dim, TYPE_STR_##TYPE, TYPE_SIZE_##TYPE, ref - 1)            @\
                                                                                                                      @\
  end function INTF_ARG_INFO(TYPE, DIM)


  interface op_arg_idx
    module procedure op_arg_idx_struct, op_arg_idx_ptr, op_arg_idx_ptr_m2
  end interface op_arg_idx

  interface op_fetch_data
    module procedure op_fetch_data_real_8, op_fetch_data_real_4, op_fetch_data_integer_4
  end interface op_fetch_data

  interface op_fetch_data_idx
    module procedure op_fetch_data_idx_real_8, op_fetch_data_idx_real_4, op_fetch_data_idx_integer_4
  end interface op_fetch_data_idx

  interface op_print_dat_to_txtfile2
    module procedure op_print_dat_to_txtfile2_real_8, op_print_dat_to_txtfile2_integer_4
  end interface op_print_dat_to_txtfile2

  interface op_reset_data_ptr
    module procedure op_reset_data_ptr_r8, op_reset_data_ptr_i4
  end interface op_reset_data_ptr

  interface op_get_data_ptr
    module procedure op_get_data_ptr_int, op_get_data_ptr_dat
  end interface op_get_data_ptr

contains

  DECL_DECL_DAT(INTEGER_4, DIM_1)
  DECL_DECL_DAT(INTEGER_4, DIM_2)
  DECL_DECL_DAT(INTEGER_4, DIM_3)
  DECL_DECL_DAT(REAL_4,    DIM_1)
  DECL_DECL_DAT(REAL_4,    DIM_2)
  DECL_DECL_DAT(REAL_4,    DIM_3)
  DECL_DECL_DAT(REAL_8,    DIM_1)
  DECL_DECL_DAT(REAL_8,    DIM_2)
  DECL_DECL_DAT(REAL_8,    DIM_3)

  DECL_DECL_DAT_OVERLAY(INTEGER_4, DIM_1)
  DECL_DECL_DAT_OVERLAY(INTEGER_4, DIM_2)
  DECL_DECL_DAT_OVERLAY(INTEGER_4, DIM_3)
  DECL_DECL_DAT_OVERLAY(REAL_4,    DIM_1)
  DECL_DECL_DAT_OVERLAY(REAL_4,    DIM_2)
  DECL_DECL_DAT_OVERLAY(REAL_4,    DIM_3)
  DECL_DECL_DAT_OVERLAY(REAL_4,    DIM_4)
  DECL_DECL_DAT_OVERLAY(REAL_4,    DIM_5)
  DECL_DECL_DAT_OVERLAY(REAL_8,    DIM_1)
  DECL_DECL_DAT_OVERLAY(REAL_8,    DIM_2)
  DECL_DECL_DAT_OVERLAY(REAL_8,    DIM_3)
  DECL_DECL_DAT_OVERLAY(REAL_8,    DIM_4)
  DECL_DECL_DAT_OVERLAY(REAL_8,    DIM_5)

  DECL_DECL_CONST(INTEGER_4, DIM_0)
  DECL_DECL_CONST(INTEGER_4, DIM_1)
  DECL_DECL_CONST(INTEGER_4, DIM_2)
  DECL_DECL_CONST(INTEGER_8, DIM_0)
  DECL_DECL_CONST(INTEGER_8, DIM_1)
  DECL_DECL_CONST(INTEGER_8, DIM_2)
  DECL_DECL_CONST(REAL_4,    DIM_0)
  DECL_DECL_CONST(REAL_4,    DIM_1)
  DECL_DECL_CONST(REAL_4,    DIM_2)
  DECL_DECL_CONST(REAL_8,    DIM_0)
  DECL_DECL_CONST(REAL_8,    DIM_1)
  DECL_DECL_CONST(REAL_8,    DIM_2)
  DECL_DECL_CONST(LOGICAL,   DIM_0)
  DECL_DECL_CONST(STRING,    DIM_0)

  DECL_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_1)
  DECL_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_1)
  DECL_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_1)
  DECL_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_2)
  DECL_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_2)
  DECL_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_2)
  DECL_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_1)
  DECL_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_1)
  DECL_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_1)
  DECL_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_1)
  DECL_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_1)
  DECL_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_2)
  DECL_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_2)
  DECL_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_2)
  DECL_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_2)
  DECL_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_2)
  DECL_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_1)
  DECL_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_1)
  DECL_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_1)
  DECL_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_1)
  DECL_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_1)
  DECL_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_2)
  DECL_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_2)
  DECL_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_2)
  DECL_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_2)
  DECL_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_2)

  DECL_OPT_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_1)
  DECL_OPT_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_1)
  DECL_OPT_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_1)
  DECL_OPT_ARG_DAT(INTEGER_4, DIM_1, MAP_DIM_2)
  DECL_OPT_ARG_DAT(INTEGER_4, DIM_2, MAP_DIM_2)
  DECL_OPT_ARG_DAT(INTEGER_4, DIM_3, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_1, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_2, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_3, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_4, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_4,    DIM_5, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_1)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_1, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_2, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_3, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_4, MAP_DIM_2)
  DECL_OPT_ARG_DAT(REAL_8,    DIM_5, MAP_DIM_2)

  DECL_ARG_GBL(INTEGER_4, DIM_0)
  DECL_ARG_GBL(INTEGER_4, DIM_1)
  DECL_ARG_GBL(INTEGER_4, DIM_2)
  DECL_ARG_GBL(REAL_4,    DIM_0)
  DECL_ARG_GBL(REAL_4,    DIM_1)
  DECL_ARG_GBL(REAL_4,    DIM_2)
  DECL_ARG_GBL(REAL_4,    DIM_3)
  DECL_ARG_GBL(REAL_8,    DIM_0)
  DECL_ARG_GBL(REAL_8,    DIM_1)
  DECL_ARG_GBL(REAL_8,    DIM_2)
  DECL_ARG_GBL(REAL_8,    DIM_3)
  DECL_ARG_GBL(LOGICAL,   DIM_0)
  DECL_ARG_GBL(LOGICAL,   DIM_1)
  DECL_ARG_GBL(LOGICAL,   DIM_2)

  DECL_OPT_ARG_GBL(INTEGER_4, DIM_0)
  DECL_OPT_ARG_GBL(INTEGER_4, DIM_1)
  DECL_OPT_ARG_GBL(INTEGER_4, DIM_2)
  DECL_OPT_ARG_GBL(REAL_4,    DIM_0)
  DECL_OPT_ARG_GBL(REAL_4,    DIM_1)
  DECL_OPT_ARG_GBL(REAL_4,    DIM_2)
  DECL_OPT_ARG_GBL(REAL_4,    DIM_3)
  DECL_OPT_ARG_GBL(REAL_8,    DIM_0)
  DECL_OPT_ARG_GBL(REAL_8,    DIM_1)
  DECL_OPT_ARG_GBL(REAL_8,    DIM_2)
  DECL_OPT_ARG_GBL(REAL_8,    DIM_3)
  DECL_OPT_ARG_GBL(LOGICAL,   DIM_0)
  DECL_OPT_ARG_GBL(LOGICAL,   DIM_1)
  DECL_OPT_ARG_GBL(LOGICAL,   DIM_2)

  DECL_ARG_INFO(INTEGER_4, DIM_0)
  DECL_ARG_INFO(INTEGER_4, DIM_1)
  DECL_ARG_INFO(INTEGER_4, DIM_2)
  DECL_ARG_INFO(REAL_4,    DIM_0)
  DECL_ARG_INFO(REAL_4,    DIM_1)
  DECL_ARG_INFO(REAL_4,    DIM_2)
  DECL_ARG_INFO(REAL_4,    DIM_3)
  DECL_ARG_INFO(REAL_8,    DIM_0)
  DECL_ARG_INFO(REAL_8,    DIM_1)
  DECL_ARG_INFO(REAL_8,    DIM_2)
  DECL_ARG_INFO(REAL_8,    DIM_3)
  DECL_ARG_INFO(LOGICAL,   DIM_0)
  DECL_ARG_INFO(LOGICAL,   DIM_1)
  DECL_ARG_INFO(LOGICAL,   DIM_2)

  subroutine op_decl_dat_temp(set, dim, type, dat, name)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(op_set) :: set
    integer(kind=c_int) :: dim

    type(op_dat) :: dat

    character(kind=c_char, len=*), optional :: name
    character(kind=c_char, len=*) :: type

    character(kind=c_char, len=:), allocatable :: name2

    character(kind=c_char, len=:), allocatable :: type2
    integer(kind=c_int) :: type_size


    if (present(name)) then
      name2 = name /@/ C_NULL_CHAR
    else
      name2 = C_CHAR_"unnamed" /@/ C_NULL_CHAR
    end if

    if (type == "r4") then
      type2 = C_CHAR_"float" /@/ C_NULL_CHAR
      type_size = 4
    else if (type == "r8") then
      type2 = C_CHAR_"double" /@/ C_NULL_CHAR
      type_size = 8
    else if (type == "i4") then
      type2 = C_CHAR_"int" /@/ C_NULL_CHAR
      type_size = 4
    else
      print *, "Error: op_del_dat_temp unknown type: ", type
      stop 1
    end if

    dat%dataCPtr = op_decl_dat_temp_char_c(set%setCPtr, dim, type2, type_size, name2)
    call c_f_pointer(dat%dataCPtr, dat%dataPtr)

  end subroutine op_decl_dat_temp

  subroutine op_arg_dat_python_check (dat, dim)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(op_dat) :: dat
    integer(kind=c_int) :: dim

    if (.not. C_ASSOCIATED(dat%dataCPtr)) then
      print *, "Error: NULL pointer in op_arg_dat"
      stop 1
    end if

    if (dat%dataPtr%dim /= dim) then
      print *, "Error: Wrong dim in op_arg_dat:", dim, dat%dataPtr%dim
      stop 1
    end if

  end subroutine

  type(op_arg) function op_arg_dat_python (dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(op_dat) :: dat
    type(op_map) :: map

    integer(kind=c_int) :: idx, dim, access
    character(kind=c_char, len=*) :: type

    integer(kind=c_int) :: c_idx
    type(c_ptr) :: c_map_ptr = C_NULL_PTR

    c_idx = idx

    if (map%mapPtr%dim /= 0) then
      c_idx = c_idx - 1
      c_map_ptr = map%mapCPtr
    end if

    call op_arg_dat_python_check(dat, dim)
    op_arg_dat_python = op_arg_dat_c(dat%dataCPtr, c_idx, c_map_ptr, dat%dataPtr%dim, &
                                     type /@/ C_NULL_CHAR, access - 1)

  end function op_arg_dat_python


  type(op_arg) function op_arg_dat_python_OP_ID (dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(op_dat) :: dat
    integer(4) :: map(2)

    integer(kind=c_int) :: idx, dim, access
    character(kind=c_char, len=*) :: type

    call op_arg_dat_python_check(dat, dim)
    op_arg_dat_python_OP_ID = op_arg_dat_c(dat%dataCPtr, idx, C_NULL_PTR, dat%dataPtr%dim, &
                                           type /@/ C_NULL_CHAR, access - 1)

  end function op_arg_dat_python_OP_ID


  type(op_arg) function op_opt_arg_dat_python (opt, dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    logical :: opt

    type(op_dat) :: dat
    type(op_map) :: map

    integer(kind=c_int) :: idx, dim, access
    character(kind=c_char, len=*) :: type

    integer(kind=c_int) :: c_idx, c_dat_dim, c_opt
    type(c_ptr) :: c_map_ptr, c_dat_ptr

    c_idx = idx
    c_dat_dim = dim
    c_map_ptr = C_NULL_PTR

    if (idx /= 0 .and. map%mapPtr%dim /= 0) then
      c_idx = c_idx - 1
      c_map_ptr = map%mapCPtr
    end if

    if (opt) then
      c_opt = 1
      c_dat_ptr = dat%dataCPtr
      c_dat_dim = dat%dataPtr%dim
    else
      c_opt = 0
      c_dat_ptr = C_NULL_PTR
      c_dat_dim = 0
    end if

    call op_arg_dat_python_check(dat, dim)
    op_opt_arg_dat_python = op_opt_arg_dat_c(c_opt, c_dat_ptr, c_idx, c_map_ptr, c_dat_dim, &
                                             type /@/ C_NULL_CHAR, access - 1)

  end function op_opt_arg_dat_python


  type(op_arg) function op_opt_arg_dat_python_OP_ID (opt, dat, idx, map, dim, type, access)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    logical :: opt

    type(op_dat) :: dat
    integer(4) :: map(2)

    integer(kind=c_int) :: idx, dim, access
    character(kind=c_char, len=*) :: type

    integer(kind=c_int) :: c_dat_dim, c_opt
    type(c_ptr) :: c_dat_ptr

    c_dat_dim = dim

    if (opt) then
      c_opt = 1
      c_dat_ptr = dat%dataCPtr
      c_dat_dim = dat%dataPtr%dim
    else
      c_opt = 0
      c_dat_ptr = C_NULL_PTR
      c_dat_dim = 0
    end if

    call op_arg_dat_python_check(dat, dim)
    op_opt_arg_dat_python_OP_ID = op_opt_arg_dat_c(c_opt, c_dat_ptr, idx, C_NULL_PTR, c_dat_dim, &
                                                   type /@/ C_NULL_CHAR, access - 1)

  end function op_opt_arg_dat_python_OP_ID


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

#ifndef OP2_ARG_POINTERS
    type (op_map_core), pointer :: idPtr
#endif
    type (op_map_core), pointer :: gblPtr

    ! calling C function
#ifndef OP2_ARG_POINTERS
    OP_ID%mapCPtr = op_decl_null_map ()
#endif
    OP_GBL%mapCPtr = op_decl_null_map ()

    ! idptr and gblPtr are required because of a gfortran compiler internal error
#ifndef OP2_ARG_POINTERS
    call c_f_pointer ( OP_ID%mapCPtr, idPtr )
#endif
    call c_f_pointer ( OP_GBL%mapCPtr, gblPtr )

#ifndef OP2_ARG_POINTERS
    idPtr%dim = 0 ! OP_ID code used in arg_set
#endif
    gblPtr%dim = -1 ! OP_GBL code used in arg_set

#ifndef OP2_ARG_POINTERS
    OP_ID%mapPtr => idPtr
#endif
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

  subroutine op_mpi_init_base( diags, global, local, base_idx)
    integer(4) :: diags
    integer(4) :: global
    integer(4) :: local
    integer(4) :: base_idx
    call op_mpi_init_base_soa( diags, global, local, base_idx, 0 )
  end subroutine op_mpi_init_base

  subroutine op_mpi_init_soa( diags, global, local, soa)
    integer(4) :: diags
    integer(4) :: global
    integer(4) :: local
    integer(4) :: soa
    call op_mpi_init_base_soa( diags, global, local, 1, soa )
  end subroutine op_mpi_init_soa

  subroutine op_mpi_init_base_soa ( diags, global, local, base_idx, soa )

    ! formal parameter
    integer(4) :: diags
    integer(4) :: global
    integer(4) :: local
    integer(4) :: base_idx
    integer(4) :: soa

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

#ifndef OP2_ARG_POINTERS
    type (op_map_core), pointer :: idPtr
#endif
    type (op_map_core), pointer :: gblPtr


    ! calling C function
#ifndef OP2_ARG_POINTERS
    OP_ID%mapCPtr = op_decl_null_map ()
#endif
    OP_GBL%mapCPtr = op_decl_null_map ()

    ! idptr and gblPtr are required because of a gfortran compiler internal error
#ifndef OP2_ARG_POINTERS
    call c_f_pointer ( OP_ID%mapCPtr, idPtr )
#endif
    call c_f_pointer ( OP_GBL%mapCPtr, gblPtr )

#ifndef OP2_ARG_POINTERS
    idPtr%dim = 0 ! OP_ID code used in arg_set
#endif
    gblPtr%dim = -1 ! OP_GBL code used in arg_set

#ifndef OP2_ARG_POINTERS
    OP_ID%mapPtr => idPtr
#endif
    OP_GBL%mapPtr => gblPtr
    call set_maps_base_c(base_idx)

    call op_mpi_init_soa_c ( 0, C_NULL_PTR, diags, global, local, soa )

    !Get the command line arguments - needs to be handled using Fortrn
    argc = command_argument_count()
    do i = 1, argc
      call get_command_argument(i, temp)
      call op_set_args_c (argc, temp) !special function to set args
    end do

  end subroutine op_mpi_init_base_soa


  subroutine op_exit ( )

    call op_exit_c (  )

  end subroutine op_exit

  subroutine op_disable_device_execution(disable)

    logical, value :: disable
    call op_disable_device_execution_c(logical(disable, kind=c_bool))

  end subroutine

  function op_check_whitelist(name) result(res)

    logical(kind=c_bool) :: res
    character(kind=c_char,len=*) :: name

    res = op_check_whitelist_c(name /@/ C_NULL_CHAR)

  end function op_check_whitelist

  subroutine op_disable_mpi_reductions(disable)

    logical, value :: disable
    call op_disable_mpi_reductions_c(logical(disable, kind=c_bool))

  end subroutine

  subroutine op_register_set(idx, set)
    integer(kind=c_int), value, intent(in) :: idx
    type(op_set) :: set

    call op_register_set_c(idx, set%setCPtr)
  end subroutine op_register_set

  type(op_set) function op_get_set( idx )
    integer(kind=c_int), value, intent(in) :: idx
    type(op_set) :: set
    set%setCPtr = op_get_set_c(idx)
    call c_f_pointer ( set%setCPtr, set%setPtr )
    op_get_set = set
  end function op_get_set


  subroutine op_decl_set ( setsize, set, opname )

    integer(kind=c_int), value, intent(in) :: setsize
    type(op_set) :: set
    character(kind=c_char,len=*), optional :: opName

    if ( present ( opname ) ) then
      set%setCPtr = op_decl_set_c ( setsize, opname/@/char(0) )
    else
      set%setCPtr = op_decl_set_c ( setsize, C_CHAR_'NONAME'/@/C_NULL_CHAR )
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
      map%mapCPtr = op_decl_map_c ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), opname/@/C_NULL_CHAR )
    else
      map%mapCPtr = op_decl_map_c ( from%setCPtr, to%setCPtr, mapdim, c_loc ( dat ), C_CHAR_'NONAME'/@/C_NULL_CHAR )
    end if

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( map%mapCPtr, map%mapPtr )

  end subroutine op_decl_map


  integer(kind=c_int) function op_free_dat_temp (dat)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(op_dat) :: dat
    op_free_dat_temp = op_free_dat_temp_c ( dat%dataCPtr )
  end function op_free_dat_temp

  INTEGER function op_get_size (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size = op_get_size_c ( set%setCPtr )

  end function op_get_size

  INTEGER function op_get_global_set_offset (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_global_set_offset = op_get_global_set_offset_c ( set%setCPtr )

  end function op_get_global_set_offset

  INTEGER function op_get_size_local_core (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size_local_core = op_get_size_local_core_c ( set%setCPtr )

  end function op_get_size_local_core

  INTEGER function op_get_size_local (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size_local = op_get_size_local_c ( set%setCPtr )

  end function op_get_size_local

  INTEGER function op_get_size_local_exec (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size_local_exec = op_get_size_local_exec_c ( set%setCPtr )

  end function op_get_size_local_exec

  INTEGER function op_get_size_local_full (set )

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set) :: set

    op_get_size_local_full = op_get_size_local_full_c ( set%setCPtr )

  end function op_get_size_local_full

  integer(8) function op_get_g_index(set)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(op_set), intent(in) :: set

    op_get_g_index = op_get_g_index_c(set%setCPtr)

  end function op_get_g_index

  type(op_arg) function op_arg_idx_struct(idx, map)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    integer(kind=c_int) :: idx
    type(op_map) :: map
    op_arg_idx_struct = op_arg_idx_c(idx-1,map%mapCPtr)
  end function op_arg_idx_struct

  type(op_arg) function op_arg_idx_ptr(idx, map)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    integer(kind=c_int) :: idx
    integer(4), dimension(*), intent(in), target :: map
    op_arg_idx_ptr = op_arg_idx_ptr_c(idx-1,c_loc(map))
  end function op_arg_idx_ptr

  type(op_arg) function op_arg_idx_ptr_m2(idx, map)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    integer(kind=c_int) :: idx
    integer(4), dimension(:,:), intent(in), target :: map
    op_arg_idx_ptr_m2 = op_arg_idx_ptr_c(idx-1,c_loc(map))
  end function op_arg_idx_ptr_m2
 
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

  subroutine op_timing2_start(name)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(kind=c_char, len=*) :: name
    call op_timing2_start_c(name /@/ C_NULL_CHAR)

  end subroutine

  subroutine op_timing2_enter(name)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(kind=c_char, len=*) :: name
    call op_timing2_enter_c(name /@/ C_NULL_CHAR)

  end subroutine

  subroutine op_timing2_enter_kernel(name, target, variant)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(kind=c_char, len=*) :: name, target, variant
    call op_timing2_enter_kernel_c(name /@/ C_NULL_CHAR, target /@/ C_NULL_CHAR, variant /@/ C_NULL_CHAR)

  end subroutine

  subroutine op_timing2_next(name)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(kind=c_char, len=*) :: name
    call op_timing2_next_c(name /@/ C_NULL_CHAR)

  end subroutine

  subroutine op_timing2_output_json(filename)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(kind=c_char, len=*) :: filename
    call op_timing2_output_json_c(filename /@/ C_NULL_CHAR)

  end subroutine

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

  subroutine op_print_dat_to_txtfile2_real_8 (dat, fileName)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(len=*) :: fileName
    real*8, dimension(*), target :: dat

    call op_print_dat_to_txtfile2_c (c_loc(dat), fileName)

  end subroutine

  subroutine op_print_dat_to_txtfile2_integer_4 (dat, fileName)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    character(len=*) :: fileName
    integer*4, dimension(*), target :: dat

    call op_print_dat_to_txtfile2_c (c_loc(dat), fileName)

  end subroutine

  subroutine op_mpi_rank (rank)

    integer(kind=c_int) :: rank

    call op_mpi_rank_c (rank)

  end subroutine op_mpi_rank

  function op_mpi_get_data(dat)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(c_ptr) :: dat
    type(op_dat) :: op_mpi_get_data

    op_mpi_get_data%dataCPtr = op_mpi_get_data_c(dat)
    call c_f_pointer(op_mpi_get_data%dataCPtr, op_mpi_get_data%dataPtr)

  end function op_mpi_get_data

  subroutine op_mpi_free_data(dat)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    type(op_dat) :: dat
    call op_mpi_free_data_c(dat%dataCPtr)

  end subroutine op_mpi_free_data

  subroutine op_print (line)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    character(kind=c_char,len=*) :: line

    call op_print_c (line/@/C_NULL_CHAR)

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

    handle%exportCptr = op_export_init_c ( nprocs, c_loc(proclist_ptr), cells2Nodes%mapPtr, &
     &   sp_nodes%setPtr, coords%dataPtr, mark%dataPtr)

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

  ! get the pointer of the data held in an op_dat
  INTEGER(8) function op_get_data_ptr_dat(dat)
    use, intrinsic :: ISO_C_BINDING
    type(op_dat)         :: dat
    op_get_data_ptr_dat = op_get_data_ptr_c( dat%dataPtr)

  end function op_get_data_ptr_dat

  INTEGER(8) function op_get_data_ptr_int(dat)
    use, intrinsic :: ISO_C_BINDING
    integer(8)         :: dat
    op_get_data_ptr_int = op_get_data_ptr_int_c( dat )

  end function op_get_data_ptr_int

  ! get the pointer of the data held in an op_dat (via the original pointer) - r8
  INTEGER(8) function op_reset_data_ptr_r8(data,mode)
    use, intrinsic :: ISO_C_BINDING
    real(8), dimension(*), target :: data
    integer(kind=c_int), value :: mode
    op_reset_data_ptr_r8 = op_reset_data_ptr_c(c_loc(data),mode)

  end function op_reset_data_ptr_r8

  ! get the pointer of the data held in an op_dat (via the original pointer) - i4
  INTEGER(8) function op_reset_data_ptr_i4(data,mode)
    use, intrinsic :: ISO_C_BINDING
    integer(4), dimension(*), target :: data
    integer(kind=c_int), value :: mode
    op_reset_data_ptr_i4 = op_reset_data_ptr_c(c_loc(data),mode)

  end function op_reset_data_ptr_i4

  ! get the pointer of the data held in an op_map
  INTEGER(8) function op_get_map_ptr(map)
    use, intrinsic :: ISO_C_BINDING
    type(op_map)         :: map
    op_get_map_ptr = op_get_map_ptr_c( map%mapPtr)

  end function op_get_map_ptr

    ! get the pointer of the map held in an op_dat (via the original pointer) - i4
  INTEGER(8) function op_reset_map_ptr(map)
    use, intrinsic :: ISO_C_BINDING
    integer(4), dimension(*), target :: map
    op_reset_map_ptr = op_reset_map_ptr_c(c_loc(map))

  end function op_reset_map_ptr


  INTEGER(8) function op_copy_map_to_fort(map)
    use, intrinsic :: ISO_C_BINDING
    integer(4), dimension(*), intent(in), target :: map
    op_copy_map_to_fort = op_copy_map_to_fort_c(c_loc(map))

  end function op_copy_map_to_fort

end module OP2_Fortran_Declarations

