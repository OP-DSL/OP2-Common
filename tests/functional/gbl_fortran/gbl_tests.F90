! Not intended to be used with OP_NO_REALLOC flag

program gbl_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use gbl_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: gbl_size = 173

  type(op_set) :: set
  type(op_dat) :: dat1, dat5, dat_iota1, dat_iota5
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  real(8), dimension(:), allocatable, target :: data1
  real(8), dimension(:), allocatable, target :: data5
  real(8), dimension(:), allocatable :: fetched1
  real(8), dimension(:), allocatable :: fetched5

  real(8), target :: g_read1
  real(8), dimension(5), target :: g_read5
  real(8), target :: g_inc1
  real(8), dimension(5), target :: g_inc5
  real(8), target :: g_min1
  real(8), dimension(5), target :: g_min5
  real(8), target :: g_max1
  real(8), dimension(5), target :: g_max5

  real(8), dimension(5) :: expected5
  real(8) :: expected
  integer :: i, d
  integer :: local_size
  integer :: my_rank
  integer :: comm_size
  integer :: local_start
#ifdef USE_MPI
  integer :: ierr
#endif

  call op_init_base(0, 0)
  call op_profile_start("FortranGblArgTests")

  call get_rank_and_size(my_rank, comm_size)

  local_size = compute_local_size(gbl_size, comm_size, my_rank)
  local_start = get_local_start(gbl_size, comm_size, my_rank)

  call op_decl_set(local_size, set, "my_set")
  write(*,*) "set size =", set%setPtr%size

  allocate(data1(local_size))
  allocate(data5(local_size * 5))
  data1 = 0.0d0
  data5 = 0.0d0

  call op_decl_dat(set, 1, "real(8)", data1, dat1, "dat1")
  call op_decl_dat(set, 5, "real(8)", data5, dat5, "dat5")

  do i = 1, local_size
    data1(i) = real(local_start + (i - 1), 8)
  end do
  do i = 1, local_size
    do d = 1, 5
      data5((i - 1) * 5 + d) = real((local_start + (i - 1)) * 5 + (d - 1), 8)
    end do
  end do

  call op_decl_dat(set, 1, "real(8)", data1, dat_iota1, "dat_iota1")
  call op_decl_dat(set, 5, "real(8)", data5, dat_iota5, "dat_iota5")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)

  ! --- READ ---
  g_read1 = 20.0d0
  call op_par_loop_2(read1_k, set, &
    op_arg_dat(dat1, -1, OP_ID, 1, "real(8)", OP_WRITE), &
    op_arg_gbl(g_read1, 1, "real(8)", OP_READ))

  allocate(fetched1(local_size))
  call op_fetch_data(dat1, fetched1)
  do i = 1, local_size
    call check(abs(fetched1(i) - g_read1) < tol, i - 1, "read1 failed")
  end do
  write(*,*) "read1 passed"

  g_read5 = (/ 30.0d0, 40.0d0, 50.0d0, 60.0d0, 70.0d0 /)
  call op_par_loop_2(read5_k, set, &
    op_arg_dat(dat5, -1, OP_ID, 5, "real(8)", OP_WRITE), &
    op_arg_gbl(g_read5, 5, "real(8)", OP_READ))

  allocate(fetched5(local_size * 5))
  call op_fetch_data(dat5, fetched5)
  do i = 1, local_size
    do d = 1, 5
      call check(abs(fetched5((i - 1) * 5 + d) - g_read5(d)) < tol, &
        (i - 1) * 5 + (d - 1), "read5 failed")
    end do
  end do
  write(*,*) "read5 passed"

  ! --- INC ---
  g_inc1 = 0.0d0
  call op_par_loop_2(inc1_k, set, &
    op_arg_dat(dat_iota1, -1, OP_ID, 1, "real(8)", OP_READ), &
    op_arg_gbl(g_inc1, 1, "real(8)", OP_INC))

  expected = real((gbl_size - 1) * gbl_size, 8) / 2.0d0
  call check(abs(g_inc1 - expected) < tol, 0, "inc1 failed")
  write(*,*) "inc1 passed"

  g_inc5 = 0.0d0
  call op_par_loop_2(inc5_k, set, &
    op_arg_dat(dat_iota5, -1, OP_ID, 5, "real(8)", OP_READ), &
    op_arg_gbl(g_inc5, 5, "real(8)", OP_INC))

  expected5 = 0.0d0
  do i = 0, gbl_size - 1
    do d = 1, 5
      expected5(d) = expected5(d) + real(i * 5 + (d - 1), 8)
    end do
  end do
  do d = 1, 5
    call check(abs(g_inc5(d) - expected5(d)) < tol, d - 1, "inc5 failed")
  end do
  write(*,*) "inc5 passed"

  ! --- MIN ---
  g_min1 = huge(g_min1)
  call op_par_loop_2(min1_k, set, &
    op_arg_dat(dat_iota1, -1, OP_ID, 1, "real(8)", OP_READ), &
    op_arg_gbl(g_min1, 1, "real(8)", OP_MIN))

  expected = 0.0d0
  call check(abs(g_min1 - expected) < tol, 0, "min1 failed")
  write(*,*) "min1 passed"

  g_min5 = (/ huge(0.0d0), huge(0.0d0), 0.4d0, huge(0.0d0), huge(0.0d0) /)
  call op_par_loop_2(min5_k, set, &
    op_arg_dat(dat_iota5, -1, OP_ID, 5, "real(8)", OP_READ), &
    op_arg_gbl(g_min5, 5, "real(8)", OP_MIN))

  expected5 = (/ 0.0d0, 1.0d0, 0.4d0, 3.0d0, 4.0d0 /)
  do d = 1, 5
    call check(abs(g_min5(d) - expected5(d)) < tol, d - 1, "min5 failed")
  end do
  write(*,*) "min5 passed"

  ! --- MAX ---
  g_max1 = -huge(g_max1)
  call op_par_loop_2(max1_k, set, &
    op_arg_dat(dat_iota1, -1, OP_ID, 1, "real(8)", OP_READ), &
    op_arg_gbl(g_max1, 1, "real(8)", OP_MAX))

  expected = real(gbl_size - 1, 8)
  call check(abs(g_max1 - expected) < tol, 0, "max1 failed")
  write(*,*) "max1 passed"

  g_max5 = (/ -huge(0.0d0), -huge(0.0d0), 1000000000.4d0, -huge(0.0d0), -huge(0.0d0) /)
  call op_par_loop_2(max5_k, set, &
    op_arg_dat(dat_iota5, -1, OP_ID, 5, "real(8)", OP_READ), &
    op_arg_gbl(g_max5, 5, "real(8)", OP_MAX))

  expected5(1) = real(gbl_size * 5 - 5, 8)
  expected5(2) = real(gbl_size * 5 - 4, 8)
  expected5(3) = 1000000000.4d0
  expected5(4) = real(gbl_size * 5 - 2, 8)
  expected5(5) = real(gbl_size * 5 - 1, 8)
  do d = 1, 5
    call check(abs(g_max5(d) - expected5(d)) < tol, d - 1, "max5 failed")
  end do
  write(*,*) "max5 passed"

  call op_profile_end()
  
  if (op_is_root() == 1) print *
    call op_profile_output()

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

  subroutine get_rank_and_size(rank, size)
    integer, intent(out) :: rank
    integer, intent(out) :: size
#ifdef USE_MPI
    integer :: ierr
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    write(*,*) "MPI rank", rank, "of", size
#else
    rank = 0
    size = 1
#endif
  end subroutine get_rank_and_size

  integer function compute_local_size(global_size, mpi_comm_size, mpi_rank)
    integer, intent(in) :: global_size
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: mpi_rank
    integer :: base, remainder

    base = global_size / mpi_comm_size
    remainder = mod(global_size, mpi_comm_size)
    compute_local_size = base
    if (mpi_rank < remainder) compute_local_size = compute_local_size + 1
  end function compute_local_size

  integer function get_local_start(global_size, mpi_comm_size, mpi_rank)
    integer, intent(in) :: global_size
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: mpi_rank
    integer :: base, remainder

    base = global_size / mpi_comm_size
    remainder = mod(global_size, mpi_comm_size)
    get_local_start = mpi_rank * base + min(mpi_rank, remainder)
  end function get_local_start

end program gbl_tests_fortran
