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

! This file defines the module used by all hdf5 OP2 back-ends (e.g. CUDA and openmp)
! and it makes use of the proper implementation in C
! (e.g. op_cuda_decl.c or op_openmp_decl.cpp)
!
! It defines the interoperable data types between OP2 C and Fortran
! and it defines the Fortran interface for declaration routines

module OP2_Fortran_hdf5_Declarations

  use OP2_Fortran_Declarations
  use, intrinsic :: ISO_C_BINDING
#ifdef OP2_WITH_CUDAFOR
  use cudafor
#endif

  interface

    type(c_ptr) function op_decl_set_hdf5_c (fileName, setName) BIND(C,name='op_decl_set_hdf5')

      use, intrinsic :: ISO_C_BINDING

      character(kind=c_char,len=1), intent(in) :: fileName
      character(kind=c_char,len=1), intent(in) :: setName

    end function op_decl_set_hdf5_c

    type(c_ptr) function op_decl_map_hdf5_c (from, to, dim, fileName, mapName) BIND(C,name='op_decl_map_hdf5')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: from, to
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1), intent(in) :: fileName
      character(kind=c_char,len=1), intent(in) :: mapName

    end function op_decl_map_hdf5_c

    type(c_ptr) function op_decl_dat_hdf5_c (set, dim, type, fileName, datName) BIND(C,name='op_decl_dat_hdf5')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: set
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1), intent(in) :: type
      character(kind=c_char,len=1), intent(in) :: fileName
      character(kind=c_char,len=1), intent(in) :: datName

    end function op_decl_dat_hdf5_c

  end interface

  interface op_decl_set_hdf5
    module procedure op_decl_set_hdf5_noSetSize, op_decl_set_hdf5_setSize
  end interface op_decl_set_hdf5

contains

  subroutine op_decl_set_hdf5_noSetSize ( set, fileName, setName )

    type(op_set) :: set
    character(kind=c_char,len=*) :: fileName
    character(kind=c_char,len=*) :: setName

    ! assume names are /0 terminated
    set%setCPtr = op_decl_set_hdf5_c (fileName, setName)

    ! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
    call c_f_pointer ( set%setCPtr, set%setPtr )

  end subroutine op_decl_set_hdf5_noSetSize

  subroutine op_decl_set_hdf5_setSize ( setSize, set, fileName, setName )

    integer(kind=c_int) :: setSize
    type(op_set) :: set
    character(kind=c_char,len=*) :: fileName
    character(kind=c_char,len=*) :: setName
    !print *,"setName ",setName
    ! assume names are /0 terminated
    set%setCPtr = op_decl_set_hdf5_c (fileName, setName)

    ! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
    call c_f_pointer ( set%setCPtr, set%setPtr )

    setSize = set%setPtr%size

  end subroutine op_decl_set_hdf5_setSize

  subroutine op_decl_map_hdf5 ( from, to, mapdim, map, fileName, mapName )

    type(op_set), intent(in) :: from, to
    integer, intent(in) :: mapdim
    type(op_map) :: map
    character(kind=c_char,len=*) :: fileName
    character(kind=c_char,len=*) :: mapName

    ! assume names are /0 terminated - will fix this if needed later
    map%mapCPtr = op_decl_map_hdf5_c ( from%setCPtr, to%setCPtr, mapdim, fileName, mapName )

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( map%mapCPtr, map%mapPtr )

  end subroutine op_decl_map_hdf5

  subroutine op_decl_dat_hdf5 ( set, datdim, data, type, fileName, datName )
    implicit none

    type(op_set), intent(in) :: set
    integer, intent(in) :: datdim
    type(op_dat) :: data
    character(kind=c_char,len=*) :: type
    character(kind=c_char,len=*) :: fileName
    character(kind=c_char,len=*) :: datName

    ! assume names are /0 terminated
    data%dataCPtr = op_decl_dat_hdf5_c ( set%setCPtr, datdim, type, fileName, datName)

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( data%dataCPtr, data%dataPtr )

    ! debugging

  end subroutine op_decl_dat_hdf5

end module OP2_Fortran_hdf5_Declarations
