! This module defines the Fortran interface towards C reference implementation
! of op_par_loop functions

#include "../../include/op2_reference_macros.h"

module OP2_Fortran_Reference

  use OP2_Fortran_Declarations

  interface

! Generation of interfaces to op_par_loop functions in C through preprocessor macros
#define ARG_LIST(N) COMMA_LIST(N, ARGS)
#define ARGS(x) data##x, itemSel##x, map##x, access##x &@

#define ARGT_LIST(N) MAP(N, ARGT)
#define ARGT(x) type(op_dat) :: data##x @ integer(kind=c_int), value :: itemSel##x, access##x @ type(op_map) :: map##x @

#ifdef PRINT_OUTPUT_DAT

#define OP_LOOP(N) subroutine op_par_loop_##N##_f(kernel, kernelName, set, &@\
   ARG_LIST(N) \
) BIND(C,name="op_par_loop_dump_"@#N) @\
   use, intrinsic :: ISO_C_BINDING @\
   import :: op_set, op_map, op_dat @\
interface @\
   subroutine kernel () BIND(C) @\
   end subroutine kernel @\
end interface @\
   character(len=*) :: kernelName @\
   type(op_set) :: set @\
   ARGT_LIST(N) \
end subroutine op_par_loop_##N##_f @

#else

#define OP_LOOP(N) subroutine op_par_loop_##N##_f(kernel, set, &@\
   ARG_LIST(N) \
) BIND(C,name="op_par_loop_"@#N) @\
   use, intrinsic :: ISO_C_BINDING @\
   import :: op_set, op_map, op_dat @\
interface @\
   subroutine kernel () BIND(C) @\
   end subroutine kernel @\
end interface @\
   type(op_set) :: set @\
   ARGT_LIST(N) \
end subroutine op_par_loop_##N##_f @

#endif

OP_LOOP(1)  OP_LOOP(2)  OP_LOOP(3)  OP_LOOP(4)  OP_LOOP(5)  OP_LOOP(6)  OP_LOOP(7)  OP_LOOP(8)  OP_LOOP(9)  OP_LOOP(10)
OP_LOOP(11) OP_LOOP(12) OP_LOOP(13) OP_LOOP(14) OP_LOOP(15) OP_LOOP(16) OP_LOOP(17) OP_LOOP(18) OP_LOOP(19) OP_LOOP(20)
OP_LOOP(21) OP_LOOP(22) OP_LOOP(23) OP_LOOP(24) OP_LOOP(25) OP_LOOP(26) OP_LOOP(27) OP_LOOP(28) OP_LOOP(29) OP_LOOP(30)
OP_LOOP(31) OP_LOOP(32) OP_LOOP(33) OP_LOOP(34) OP_LOOP(35) OP_LOOP(36) OP_LOOP(37) OP_LOOP(38) OP_LOOP(39) OP_LOOP(40)

end interface

  contains

#define CARG_LIST(N) MAP(N, CARG)
#define CARG(x) integer(kind=c_int) :: itemSelC##x @

#define CSET_LIST(N) MAP(N, CSET)
#define CSET(x) itemSelC##x = itemSel##x - 1@

#define ARG_LISTK(N) COMMA_LIST(N, ARGSK)
#define ARGSK(x) data##x, itemSelC##x, map##x, access##x &@

#ifdef PRINT_OUTPUT_DAT

#define OP_LOOP2(N) subroutine op_par_loop_##N(kernel, kernelName, set, &@\
   ARG_LIST(N) \
) @\
   external kernel @\
   type(op_set) :: set @\
   character(len=*) :: kernelName @\
   ARGT_LIST(N) \
   CARG_LIST(N) \
   CSET_LIST(N) \
   call op_par_loop_##N##_f(kernel, kernelName, set, &@\
      ARG_LISTK(N) \
   ) @\
end subroutine op_par_loop_##N @

#else

! ARG_LIST and ARGT_LIST reused from OP_LOOP
#define OP_LOOP2(N) subroutine op_par_loop_##N(kernel, set, &@\
   ARG_LIST(N) \
) @\
   external kernel @\
   type(op_set) :: set @\
   ARGT_LIST(N) \
   CARG_LIST(N) \
   CSET_LIST(N) \
   call op_par_loop_##N##_f(kernel, set, &@\
      ARG_LISTK(N) \
   ) @\
end subroutine op_par_loop_##N @

#endif

OP_LOOP2(1)  OP_LOOP2(2)  OP_LOOP2(3)  OP_LOOP2(4)  OP_LOOP2(5)  OP_LOOP2(6)  OP_LOOP2(7)  OP_LOOP2(8)  OP_LOOP2(9)  OP_LOOP2(10)
OP_LOOP2(11) OP_LOOP2(12) OP_LOOP2(13) OP_LOOP2(14) OP_LOOP2(15) OP_LOOP2(16) OP_LOOP2(17) OP_LOOP2(18) OP_LOOP2(19) OP_LOOP2(20)
OP_LOOP2(21) OP_LOOP2(22) OP_LOOP2(23) OP_LOOP2(24) OP_LOOP2(25) OP_LOOP2(26) OP_LOOP2(27) OP_LOOP2(28) OP_LOOP2(29) OP_LOOP2(30)
OP_LOOP2(31) OP_LOOP2(32) OP_LOOP2(33) OP_LOOP2(34) OP_LOOP2(35) OP_LOOP2(36) OP_LOOP2(37) OP_LOOP2(38) OP_LOOP2(39) OP_LOOP2(40)


end module OP2_Fortran_Reference

