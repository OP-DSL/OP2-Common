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

! Implementation of serial reference op_par_loop functions

#include "../include/op2_reference_macros.h"

! FORTRAN interface of C-based reference implementation
module OP2_Fortran_Reference

  use OP2_Fortran_Declarations

  interface

! Generation of interfaces to op_par_loop functions in C through preprocessor macros
#define ARG_LIST(N) COMMA_LIST(N, ARGS)
#define ARGS(x) arg##x &@

#define ARGT_LIST(N) MAP(N, ARGT)
#define ARGT(x) type(op_arg) :: arg##x @

#define OP_LOOP(N) subroutine op_par_loop_##N##_f(kernel, set, &@\
   ARG_LIST(N) \
) BIND(C,name="op_par_loop_"@#N) @\
   use, intrinsic :: ISO_C_BINDING @\
   import :: op_set_core, op_arg @\
interface @\
   subroutine kernel () BIND(C) @\
   end subroutine kernel @\
end interface @\
   type(op_set_core) :: set @\
   ARGT_LIST(N) @\
end subroutine op_par_loop_##N##_f @

OP_LOOP(1)  OP_LOOP(2)  OP_LOOP(3)  OP_LOOP(4)  OP_LOOP(5)  OP_LOOP(6)  OP_LOOP(7)  OP_LOOP(8)  OP_LOOP(9)  OP_LOOP(10)
OP_LOOP(11) OP_LOOP(12) OP_LOOP(13) OP_LOOP(14) OP_LOOP(15) OP_LOOP(16) OP_LOOP(17) OP_LOOP(18) OP_LOOP(19) OP_LOOP(20)
OP_LOOP(21) OP_LOOP(22) OP_LOOP(23) OP_LOOP(24) OP_LOOP(25) OP_LOOP(26) OP_LOOP(27) OP_LOOP(28) OP_LOOP(29) OP_LOOP(30)
OP_LOOP(31) OP_LOOP(32) OP_LOOP(33) OP_LOOP(34) OP_LOOP(35) OP_LOOP(36) OP_LOOP(37) OP_LOOP(38) OP_LOOP(39) OP_LOOP(40)
OP_LOOP(41) OP_LOOP(42) OP_LOOP(43)

end interface

  contains

#define ARG_NOCORE_LIST(N) MAP(N, ARG_NOCORE)
#define ARG_NOCORE(x) type(op_arg) :: arg##x @

! ARG_LIST and ARGT_LIST reused from OP_LOOP
#define OP_LOOP2(N) subroutine op_par_loop_##N(kernel, set, &@\
   ARG_LIST(N) \
) @\
   external kernel @\
   type(op_set) :: set @\
   ARG_NOCORE_LIST(N) @\
   call op_par_loop_##N##_f(kernel, set%setPtr, &@\
      ARG_LIST(N) \
   ) @\
end subroutine op_par_loop_##N @

OP_LOOP2(1)  OP_LOOP2(2)  OP_LOOP2(3)  OP_LOOP2(4)  OP_LOOP2(5)  OP_LOOP2(6)  OP_LOOP2(7)  OP_LOOP2(8)  OP_LOOP2(9)  OP_LOOP2(10)
OP_LOOP2(11) OP_LOOP2(12) OP_LOOP2(13) OP_LOOP2(14) OP_LOOP2(15) OP_LOOP2(16) OP_LOOP2(17) OP_LOOP2(18) OP_LOOP2(19) OP_LOOP2(20)
OP_LOOP2(21) OP_LOOP2(22) OP_LOOP2(23) OP_LOOP2(24) OP_LOOP2(25) OP_LOOP2(26) OP_LOOP2(27) OP_LOOP2(28) OP_LOOP2(29) OP_LOOP2(30)
OP_LOOP2(31) OP_LOOP2(32) OP_LOOP2(33) OP_LOOP2(34) OP_LOOP2(35) OP_LOOP2(36) OP_LOOP2(37) OP_LOOP2(38) OP_LOOP2(39) OP_LOOP2(40)
OP_LOOP2(41) OP_LOOP2(42) OP_LOOP2(43)

end module OP2_Fortran_Reference
