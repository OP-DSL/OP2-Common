module airfoil_seq

USE OP2_FORTRAN_JIT
use OP2_CONSTANTS
use, intrinsic :: iso_c_binding

type(c_funptr) :: proc_addr_res_calc
type(c_funptr) :: proc_addr_adt_calc

type(c_ptr) :: handle

 contains
#include "bres_calc.inc"
#include "res_calc.inc"
#include "adt_calc.inc"
#include "save_soln.inc"
#include "update.inc"


subroutine jit_compile ()
use, intrinsic :: iso_c_binding

IMPLICIT NONE

integer(c_int), parameter :: RTLD_LAZY=1 ! value extracte from the C header file
integer STATUS

  ! compile *_seqkernel_rec.F90 using system command
  write(*,*) 'JIT compiling procedure res_calc_host_rec'
  call execute_command_line ("make -j genseq_jit", exitstat=STATUS)

  ! dynamically load airfoil_seqkernel_rec.so
  handle=dlopen("./airfoil_seqkernel_rec.so"//c_null_char, RTLD_LAZY)
  if (.not. c_associated(handle))then
    print*, 'Unable to load DLL ./airfoil_seqkernel_rec.so'
    stop
  end if

  if(.not. c_associated(proc_addr_res_calc)) then
    proc_addr_res_calc=dlsym(handle, "res_calc_module_execute_mp_res_calc_host_rec_"//c_null_char)
    if (.not. c_associated(proc_addr_res_calc))then
      write(*,*) 'Unable to load the procedure res_calc_module_execute_mp_res_calc_host_rec_'
      stop
    end if
  end if
  if(.not. c_associated(proc_addr_adt_calc)) then
    proc_addr_adt_calc=dlsym(handle, "adt_calc_module_execute_mp_adt_calc_host_rec_"//c_null_char)
    if (.not. c_associated(proc_addr_adt_calc))then
      write(*,*) 'Unable to load the procedure adt_calc_module_execute_mp_adt_calc_host_rec_'
      stop
    end if
  end if

  JIT_COMPILED = .true.

end subroutine jit_compile


end module airfoil_seq
