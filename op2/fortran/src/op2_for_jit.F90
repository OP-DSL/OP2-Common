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
! It defines the interfaces required for JIT compilation with the Fortran API

module OP2_Fortran_JIT

  use, intrinsic :: ISO_C_BINDING

  !
  ! interface to linux API
  !
  interface
      function dlopen(filename,mode) bind(c,name="dlopen")
          ! void *dlopen(const char *filename, int mode);
          use iso_c_binding
          implicit none
          type(c_ptr) :: dlopen
          character(c_char), intent(in) :: filename(*)
          integer(c_int), value :: mode
      end function

      function dlsym(handle,name) bind(c,name="dlsym")
          ! void *dlsym(void *handle, const char *name);
          use iso_c_binding
          implicit none
          type(c_funptr) :: dlsym
          type(c_ptr), value :: handle
          character(c_char), intent(in) :: name(*)
      end function

      function dlclose(handle) bind(c,name="dlclose")
          ! int dlclose(void *handle);
          use iso_c_binding
          implicit none
          integer(c_int) :: dlclose
          type(c_ptr), value :: handle
      end function
  end interface

end module OP2_Fortran_JIT