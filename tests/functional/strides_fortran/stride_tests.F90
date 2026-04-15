! Not intended to be used with OP_NO_REALLOC flag

program stride_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use stride_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: gbl_size = 201

  type(op_set) :: set
  type(op_dat) :: dat5_0, dat5_1, dat_iota1
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  real(8), dimension(:), allocatable, target :: data5_0
  real(8), dimension(:), allocatable, target :: data5_1
  real(8), dimension(:), allocatable, target :: data1
  real(8), dimension(:), allocatable :: fetched5_0
  real(8), dimension(:), allocatable :: fetched5_1

  integer :: local_size
  integer :: my_rank
  integer :: comm_size
  integer :: local_start
  integer :: i, d
  real(8) :: expected0, expected1

  call op_init_base(0, 0)

  call get_rank_and_size(my_rank, comm_size)

  local_size = compute_local_size(gbl_size, comm_size, my_rank)
  local_start = get_local_start(gbl_size, comm_size, my_rank)

  call op_decl_set(local_size, set, "my_set")
  write(*,*) "local_size =", local_size, ", set size =", set%setPtr%size

  allocate(data5_0(local_size * 5))
  allocate(data5_1(local_size * 5))
  allocate(data1(local_size))

  data5_0 = 0.0d0
  data5_1 = 0.0d0

  do i = 1, local_size
    data1(i) = real(local_start + (i - 1), 8)
  end do

  call op_decl_dat(set, 5, "real(8)", data5_0, dat5_0, "dat5_0")
  call op_decl_dat(set, 5, "real(8)", data5_1, dat5_1, "dat5_1")
  call op_decl_dat(set, 1, "real(8)", data1, dat_iota1, "dat_iota1")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)

  ! --- Regular Stride Tests ---
  call op_par_loop_2(write5, set, &
    op_arg_dat(dat5_0, -1, OP_ID, 5, "real(8)", OP_WRITE), &
    op_arg_dat(dat_iota1, -1, OP_ID, 1, "real(8)", OP_READ))

  allocate(fetched5_0(local_size * 5))
  call op_fetch_data(dat5_0, fetched5_0)
  do i = 1, local_size
    do d = 1, 5
      expected0 = real(local_start + (i - 1), 8) * real(d, 8) * (10.0d0 ** (d - 1))
      call check(abs(fetched5_0((i - 1) * 5 + d) - expected0) < tol, &
        (i - 1) * 5 + (d - 1), "write5 failed")
    end do
  end do
  write(*,*) "write5 passed"

  ! --- Function call within kernel Stride Tests ---
  call op_par_loop_3(write5_within_kernel, set, &
    op_arg_dat(dat5_0, -1, OP_ID, 5, "real(8)", OP_WRITE), &
    op_arg_dat(dat5_1, -1, OP_ID, 5, "real(8)", OP_WRITE), &
    op_arg_dat(dat_iota1, -1, OP_ID, 1, "real(8)", OP_READ))

  allocate(fetched5_1(local_size * 5))
  call op_fetch_data(dat5_1, fetched5_1)
  call op_fetch_data(dat5_0, fetched5_0)

  do i = 1, local_size
    do d = 1, 5
      expected0 = real(local_start + (i - 1), 8) * real(d, 8) / (10.0d0 ** (d - 1))
      expected1 = real(local_start + (i - 1), 8) * real(d, 8) * (10.0d0 ** (d - 1))
      call check(abs(fetched5_0((i - 1) * 5 + d) - expected0) < tol, &
        (i - 1) * 5 + (d - 1), "write5_within_kernel failed")
      call check(abs(fetched5_1((i - 1) * 5 + d) - expected1) < tol, &
        (i - 1) * 5 + (d - 1), "write5_within_kernel failed")
    end do
  end do
  write(*,*) "write5_within_kernel passed"

  call op_exit()

contains ! ---------------------------------------------------------------------------------------------------
  
  ! --- Utility functions ---
  subroutine check(cond, idx, msg)
    logical, intent(in) :: cond
    integer, intent(in) :: idx
    character(len=*), intent(in) :: msg

    if (.not. cond) then
      write(*,*) "ERROR:", trim(msg), "at idx:", idx
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

end program stride_tests_fortran
