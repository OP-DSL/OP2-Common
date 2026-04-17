! Not intended to be used with OP_NO_REALLOC flag

program const_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support
  
  use const_constants
  use const_kernels
  
  use, intrinsic :: iso_c_binding

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: size = 32

  type(op_set) :: set
  type(op_dat) :: dat1, dat4
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  real(8), dimension(:), allocatable, target :: data1
  real(8), dimension(:), allocatable, target :: data4
  real(8), dimension(:), allocatable :: fetched1
  real(8), dimension(:), allocatable :: fetched4

  integer :: i, d

  call op_init_base(0, 0)

  call op_decl_set(size, set, "my_set")
  write(*,*) "set size =", set%setPtr%size

  allocate(data1(size))
  allocate(data4(size * 4))
  data1 = 0.0d0
  data4 = 0.0d0

  call op_decl_dat(set, 1, "real(8)", data1, dat1, "dat1")
  call op_decl_dat(set, 4, "real(8)", data4, dat4, "dat4")

  call op_decl_const(my_const1, 1, "real(8)")
  call op_decl_const(my_const4, 4, "real(8)")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)
  
  ! --- CONST Check DIM=1 ---
  call op_par_loop_1(consts1, set, &
    op_arg_dat(dat1, -1, OP_ID, 1, "real(8)", OP_WRITE))

  allocate(fetched1(size))
  call op_fetch_data(dat1, fetched1)
  do i = 1, size
    call check(abs(fetched1(i) - my_const1) < tol, i - 1, "consts1 failed")
  end do
  write(*,*) "consts1 passed"

  ! --- CONST Check DIM=4 ---
  call op_par_loop_1(consts4, set, &
    op_arg_dat(dat4, -1, OP_ID, 4, "real(8)", OP_WRITE))

  allocate(fetched4(size * 4))
  call op_fetch_data(dat4, fetched4)
  do i = 1, size
    do d = 1, 4
      call check(abs(fetched4((i - 1) * 4 + d) - my_const4(d)) < tol, &
        (i - 1) * 4 + (d - 1), "consts4 failed")
    end do
  end do
  write(*,*) "consts4 passed"

  call op_exit()

contains ! ---------------------------------------------------------------------------------------------------

  ! --- Utility functions ---
  subroutine check(cond, idx, msg)
    logical, intent(in) :: cond
    integer, intent(in) :: idx
    character(len=*), intent(in) :: msg

    if (.not. cond) then
      write(*,*) "ERROR:", trim(msg), " at idx:", idx
      call op_exit()
      stop 1
    end if
  end subroutine check

  subroutine nullify_dummy(set_dummy, map_dummy, dat_dummy)
    use, intrinsic :: iso_c_binding
    type(op_set), intent(inout) :: set_dummy
    type(op_map), intent(inout) :: map_dummy
    type(op_dat), intent(inout) :: dat_dummy

    nullify(set_dummy%setPtr)
    set_dummy%setCptr = c_null_ptr
    nullify(map_dummy%mapPtr)
    map_dummy%mapCptr = c_null_ptr
    nullify(dat_dummy%dataPtr)
    dat_dummy%dataCptr = c_null_ptr
  end subroutine nullify_dummy

end program const_tests_fortran
