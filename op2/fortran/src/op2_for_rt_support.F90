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

! This module defines the interoperable data structures of the OP2 RT support
! (i.e. plan struct) and defines the interface for the C plan function

module OP2_Fortran_RT_Support

  use, intrinsic :: ISO_C_BINDING

#ifdef OP2_WITH_CUDAFOR
  use cudafor
#endif

  integer(kind=c_int), parameter :: F_OP_ARG_DAT = 0
  integer(kind=c_int), parameter :: F_OP_ARG_GBL = 1

  type, BIND(C) :: op_plan

    ! input arguments
    type(c_ptr) ::         name
    type(c_ptr) ::         set
    integer(kind=c_int) :: nargs, ninds, ninds_staged, part_size
    type(c_ptr) ::         in_maps
    type(c_ptr) ::         dats
    type(c_ptr) ::         idxs
    type(c_ptr) ::         optflags
    type(c_ptr) ::         accs
    type(c_ptr) ::         inds_staged

    ! execution plan
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      nthrcol ! number of thread colors for each block
    type(c_devptr) ::      thrcol  ! thread colors
    type(c_devptr) ::      col_reord ! element permutation by color for the block
    type(c_ptr) ::         col_offsets ! offsets to element permutation by color for the block
    type(c_ptr) ::         color2_offsets ! offsets to element permutation by color for flat coloring
    type(c_ptr) ::         offset  ! offset for primary set
    type(c_devptr) ::      offset_d  ! offset for primary set
#else
    type(c_ptr) ::         nthrcol ! number of thread colors for each block
    type(c_ptr) ::         thrcol  ! thread colors
    type(c_ptr) ::         col_reord ! element permutation by color for the block
    type(c_ptr) ::         col_offsets ! offsets to element permutation by color for the block
    type(c_ptr) ::         color2_offsets ! offsets to element permutation by color for flat coloring
    type(c_ptr) ::         offset  ! offset for primary set
    type(c_ptr) ::         offset_d  ! offset for primary set
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
    type(c_ptr) ::         nelems ! number of elements in each block
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      nelems_d ! number of elements in each block
#else
    type(c_ptr) ::         nelems_d ! number of elements in each block
#endif
    integer(kind=c_int) :: ncolors_core ! mumber of core colors in MPI
    integer(kind=c_int) :: ncolors_owned ! number of colors in MPI for blocks that only have owned elements
    integer(kind=c_int) :: ncolors ! number of block colors
    type(c_ptr) ::         ncolblk  ! number of blocks for each color
    type(c_ptr) ::         blkmap ! block mapping
#ifdef OP2_WITH_CUDAFOR
    type(c_devptr) ::      blkmap_d ! block mapping
#else
    type(c_ptr) ::         blkmap_d ! block mapping
#endif

    type(c_ptr) ::         nsharedCol ! bytes of shared memory required per block colour
    integer(kind=c_int) :: nshared ! bytes of shared memory required
    real(kind=c_float) ::  transfer ! bytes of data transfer per kernel call
    real(kind=c_float) ::  transfer2 ! bytes of cache line per kernel call
    integer(kind=c_int) :: count ! number fo times called (should not work for fortran?)

  end type op_plan


  interface

    ! C wrapper to plan function for Fortran
    type(c_ptr) function FortranPlanCaller (name, set, partitionSize, argsNumber, args, indsNumber, inds, staging) &
      & BIND(C,name='FortranPlanCaller')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      character(kind=c_char) ::     name(*)    ! name of kernel
      type(c_ptr), value ::         set        ! iteration set
      integer(kind=c_int), value :: partitionSize
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args
      integer(kind=c_int), value :: indsNumber ! number of arguments accessed indirectly via a map

      ! indexes for indirectly accessed arguments (same indrectly accessed argument = same index)
      integer(kind=c_int), dimension(*) :: inds

      integer(kind=c_int), value :: staging ! What to stage: 0 - nothing, 1 - OP_INCs, 2 - all indirectly accessed data

    end function FortranPlanCaller

    subroutine op_dat_write_index_c(set, dat) BIND(C,name='op_dat_write_index')
      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations
      type(c_ptr), value ::         set,dat
    end subroutine


    integer(kind=c_int) function getSetSizeFromOpArg (arg) BIND(C,name='getSetSizeFromOpArg')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg

    end function

    integer(kind=c_int) function getMapDimFromOpArg (arg) BIND(C,name='getMapDimFromOpArg')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg

    end function

    integer(kind=c_int) function getHybridGPU () BIND(C,name='getHybridGPU')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

    end function

    integer(kind=c_int) function reductionSize (args, argsNumber) BIND(C,name='reductionSize')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end function reductionSize

    subroutine op_upload_all () BIND(C,name='op_upload_all')

      use, intrinsic :: ISO_C_BINDING

    end subroutine

    integer(kind=c_int) function op_is_root () BIND(C,name='op_is_root')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

    end function

    subroutine op_partition_c (lib_name, lib_routine, prime_set, prime_map, coords) BIND(C,name='op_partition')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      character(kind=c_char) :: lib_name(*)
      character(kind=c_char) :: lib_routine(*)

      type(op_set_core) :: prime_set
      type(op_map_core) :: prime_map
      type(op_dat_core) :: coords

    end subroutine

    subroutine op_partition_ptr_c (lib_name, lib_routine, prime_set, prime_map, coords) BIND(C,name='op_partition_ptr')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      character(kind=c_char) :: lib_name(*)
      character(kind=c_char) :: lib_routine(*)

      type(op_set_core) :: prime_set
      type(c_ptr), value, intent(in) :: prime_map
      type(c_ptr), value, intent(in) :: coords

    end subroutine


    subroutine op_renumber_c (base) BIND(C,name='op_renumber')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_map_core) :: base

    end subroutine

    subroutine op_renumber_ptr_c (ptr) BIND(C,name='op_renumber_ptr')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(c_ptr), value, intent(in) :: ptr

    end subroutine

    integer(kind=c_int) function op_mpi_halo_exchanges (set, argsNumber, args) BIND(C,name='op_mpi_halo_exchanges')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(c_ptr), value ::         set        ! iteration set
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end function op_mpi_halo_exchanges

    integer(kind=c_int) function op_mpi_halo_exchanges_grouped (set, argsNumber, args, device) BIND(C,name='op_mpi_halo_exchanges_grouped')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(c_ptr), value ::         set        ! iteration set
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args
      integer(kind=c_int), value :: device     ! 1 for CPU 2 for GPU

    end function op_mpi_halo_exchanges_grouped

    subroutine op_mpi_wait_all (argsNumber, args) BIND(C,name='op_mpi_wait_all')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine

    subroutine op_mpi_wait_all_grouped (argsNumber, args, device) BIND(C,name='op_mpi_wait_all_grouped')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args
      integer(kind=c_int), value :: device     ! 1 for CPU 2 for GPU

    end subroutine

    subroutine op_mpi_set_dirtybit (argsNumber, args) BIND(C,name='op_mpi_set_dirtybit')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine

    integer(kind=c_int) function op_mpi_halo_exchanges_cuda (set, argsNumber, args) BIND(C,name='op_mpi_halo_exchanges_cuda')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(c_ptr), value ::         set        ! iteration set
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end function op_mpi_halo_exchanges_cuda

    subroutine op_mpi_wait_all_cuda (argsNumber, args) BIND(C,name='op_mpi_wait_all_cuda')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine

    subroutine op_get_all_cuda (argsNumber, args) BIND(C,name='op_get_all_cuda')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine


    subroutine op_mpi_set_dirtybit_cuda (argsNumber, args) BIND(C,name='op_mpi_set_dirtybit_cuda')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine

    subroutine op_mpi_reduce_combined (args, argsNumber) BIND(C,name='op_mpi_reduce_combined')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg), dimension(*) :: args       ! array with op_args
      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop

   end subroutine

    subroutine op_mpi_reduce_int (arg, data) BIND(C,name='op_mpi_reduce_int')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg
      type(c_ptr) :: data

    end subroutine

    subroutine op_mpi_reduce_double (arg, data) BIND(C,name='op_mpi_reduce_double')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg
      type(c_ptr) :: data

    end subroutine

    subroutine op_mpi_reduce_float (arg, data) BIND(C,name='op_mpi_reduce_float')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg
      type(c_ptr) :: data

    end subroutine

    ! commented while waiting for C-side support
    subroutine op_mpi_reduce_bool (arg, data) BIND(C,name='op_mpi_reduce_bool')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg
      type(c_ptr) :: data

    end subroutine

    subroutine prepareScratch(args,argsNumber, nthreads) BIND(C,name='prepareScratch')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of op_dat arguments to op_par_loop
      integer(kind=c_int), value :: nthreads
      type(op_arg), dimension(*) :: args       ! array with op_args

    end subroutine

    ! debugging routines
    subroutine op_dump_arg (arg) BIND(C,name='op_dump_arg')

      use, intrinsic :: ISO_C_BINDING
      use OP2_Fortran_Declarations

      type(op_arg) :: arg

    end subroutine

    subroutine decrement_all_maps () BIND(C,name='decrement_all_mappings')
    end subroutine decrement_all_maps

    subroutine increment_all_maps () BIND(C,name='increment_all_mappings')
    end subroutine increment_all_maps

    integer(kind=c_int) function op_mpi_size () BIND(C,name='op_mpi_size')
      use, intrinsic :: ISO_C_BINDING
    end function op_mpi_size

    subroutine op_barrier () BIND(C,name='op_barrier')
    end subroutine op_barrier

    integer(kind=c_int) function setKernelTime (id, name, kernelTime, transfer, transfer2, count) BIND(C,name='setKernelTime')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value :: id
      character(kind=c_char) :: name(*)
      real(kind=c_double), value :: kernelTime
      real(kind=c_float), value :: transfer
      real(kind=c_float), value :: transfer2
      integer(kind=c_int), value :: count

    end function setKernelTime

  end interface

  contains

  subroutine op_partition (lib_name, lib_routine, prime_set, prime_map, coords)

    use, intrinsic :: ISO_C_BINDING
    use OP2_Fortran_Declarations

    implicit none

    character(kind=c_char,len=*) :: lib_name
    character(kind=c_char,len=*) :: lib_routine

    type(op_set) :: prime_set
    type(op_map) :: prime_map
    type(op_dat) :: coords

    call op_partition_c (lib_name//C_NULL_CHAR, lib_routine//C_NULL_CHAR, prime_set%setPtr, prime_map%mapPtr, coords%dataPtr)

  end subroutine

  subroutine op_partition2 (lib_name, lib_routine, prime_set, prime_map, coords)

    use, intrinsic :: ISO_C_BINDING
    use OP2_Fortran_Declarations

    implicit none

    character(kind=c_char,len=*) :: lib_name
    character(kind=c_char,len=*) :: lib_routine

    type(op_set) :: prime_set
    integer*4, dimension(*), target :: prime_map
    real*8, dimension(*), target :: coords

    call op_partition_ptr_c (lib_name//C_NULL_CHAR, lib_routine//C_NULL_CHAR, prime_set%setPtr, c_loc(prime_map), c_loc(coords))

  end subroutine

  subroutine op_renumber (base_map)

    use, intrinsic :: ISO_C_BINDING
    use OP2_Fortran_Declarations

    implicit none

    type(op_map) :: base_map

    call op_renumber_c (base_map%mapPtr)

  end subroutine
  
  subroutine op_renumber_ptr (prime_map)  
    use, intrinsic :: ISO_C_BINDING

    implicit none
    integer*4, dimension(*), target :: prime_map
    call op_renumber_ptr_c(c_loc(prime_map))
  end subroutine


  subroutine op_dat_write_index(set, dat)
    use, intrinsic :: ISO_C_BINDING
    use OP2_Fortran_Declarations

    implicit none
    type(op_set) :: set
    integer(4), dimension(*), target :: dat

    call op_dat_write_index_c(set%setCPtr, c_loc(dat))
  end subroutine

end module OP2_Fortran_RT_Support
