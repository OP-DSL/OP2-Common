! Not intended to be used with OP_NO_REALLOC flag

program idx_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use idx_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: gbl_edges = 48
  integer(4), parameter :: gbl_nodes = 64
  integer(4), parameter :: map_dim = 3

  type(op_set) :: edges, nodes
  type(op_map) :: ppedge, qqedge
  type(op_dat) :: direct_dat, indirect_dat, mixed_dat
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  integer(c_int), pointer, dimension(:) :: f_ppedge, f_qqedge

  integer(4), dimension(:), allocatable, target :: ppedge_data
  integer(4), dimension(:), allocatable, target :: qqedge_data
  real(8), dimension(:), allocatable, target :: direct_data
  real(8), dimension(:), allocatable, target :: indirect_data
  real(8), dimension(:), allocatable, target :: mixed_data
  real(8), dimension(:), allocatable :: fetched

  integer(4) :: local_edges, local_nodes, edge_start
  integer(4) :: edge_size_inc_halo
  integer(4) :: my_rank, comm_size
  integer(4) :: e, d, global_edge, base
  real(8) :: expected

  call op_init_base(0, 0)
  call op_profile_start("FortranIdxTests")

  call get_rank_and_size(my_rank, comm_size)

  local_edges = compute_local_size(gbl_edges, comm_size, my_rank)
  local_nodes = compute_local_size(gbl_nodes, comm_size, my_rank)
  edge_start = get_local_start(gbl_edges, comm_size, my_rank)

  allocate(ppedge_data(local_edges * map_dim))
  allocate(qqedge_data(local_edges * map_dim))
  do e = 0, local_edges - 1
    global_edge = edge_start + e
    do d = 0, map_dim - 1
      ppedge_data(e * map_dim + d + 1) = mod(global_edge * 7 + d * 11, gbl_nodes)
      qqedge_data(e * map_dim + d + 1) = mod(global_edge * 5 + d * 3 + 1, gbl_nodes)
    end do
  end do

  allocate(direct_data(local_edges))
  allocate(indirect_data(local_edges * map_dim))
  allocate(mixed_data(local_edges * (map_dim + 1)))
  direct_data = 0.0d0
  indirect_data = 0.0d0
  mixed_data = 0.0d0

  call op_decl_set(local_edges, edges, "edges")
  call op_decl_set(local_nodes, nodes, "nodes")
  write(*,*) "edge set size =", edges%setPtr%size, ", node set size =", nodes%setPtr%size

  call op_decl_map(edges, nodes, map_dim, ppedge_data, ppedge, "ppedge")
  call op_decl_map(edges, nodes, map_dim, qqedge_data, qqedge, "qqedge")
  call op_decl_dat(edges, 1, "real(8)", direct_data, direct_dat, "direct_dat")
  call op_decl_dat(edges, map_dim, "real(8)", indirect_data, indirect_dat, "indirect_dat")
  call op_decl_dat(edges, map_dim + 1, "real(8)", mixed_data, mixed_dat, "mixed_dat")
  
  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)

  edge_size_inc_halo = edges%setPtr%size + edges%setPtr%exec_size + edges%setPtr%nonexec_size

  call c_f_pointer(ppedge%mapPtr%map, f_ppedge, [edge_size_inc_halo * map_dim])
  call c_f_pointer(qqedge%mapPtr%map, f_qqedge, [edge_size_inc_halo * map_dim])

  ! --- Direct idx: op_arg_idx(-1, OP_ID) ---
  call op_par_loop_2(write_direct_idx, edges, &
    op_arg_dat(direct_dat, -1, OP_ID, 1, "real(8)", OP_WRITE), &
    op_arg_idx(-1, OP_ID))

  allocate(fetched(edge_size_inc_halo))
  call op_fetch_data(direct_dat, fetched)

  do e = 0, edges%setPtr%size - 1
    expected = real(e + 1, 8)
    call check(abs(fetched(e + 1) - expected) < tol, e, &
      "op_arg_idx(-1, OP_ID) failed")
  end do
  write(*,*) "direct idx passed [rank", my_rank, "]"

  deallocate(fetched)

  ! --- Indirect idx: op_arg_idx values from two maps ---
  call op_par_loop_4(write_indirect_idx, edges, &
    op_arg_dat(indirect_dat, -1, OP_ID, map_dim, "real(8)", OP_WRITE), &
    op_arg_idx(1, ppedge), &
    op_arg_idx(2, qqedge), &
    op_arg_idx(3, qqedge))

  allocate(fetched(edge_size_inc_halo * map_dim))
  call op_fetch_data(indirect_dat, fetched)

  do e = 0, edges%setPtr%size - 1
    expected = real(f_ppedge(e * map_dim + 1) + 1, 8)
    call check(abs(fetched(e * map_dim + 1) - expected) < tol, &
      e * map_dim, "op_arg_idx(1, ppedge) failed")

    expected = real(f_qqedge(e * map_dim + 2) + 1, 8)
    call check(abs(fetched(e * map_dim + 2) - expected) < tol, &
      e * map_dim + 1, "op_arg_idx(2, qqedge) failed")

    expected = real(f_qqedge(e * map_dim + 3) + 1, 8)
    call check(abs(fetched(e * map_dim + 3) - expected) < tol, &
      e * map_dim + 2, "op_arg_idx(3, qqedge) failed")
  end do
  write(*,*) "indirect idx with two maps passed [rank", my_rank, "]"

  deallocate(fetched)

  ! --- Combined direct and indirect idx in the same kernel ---
  call op_par_loop_5(write_mixed_idx, edges, &
    op_arg_dat(mixed_dat, -1, OP_ID, map_dim + 1, "real(8)", OP_WRITE), &
    op_arg_idx(-1, OP_ID), &
    op_arg_idx(1, ppedge), &
    op_arg_idx(2, ppedge), &
    op_arg_idx(3, ppedge))

  allocate(fetched(edge_size_inc_halo * (map_dim + 1)))
  call op_fetch_data(mixed_dat, fetched)

  do e = 0, edges%setPtr%size - 1
    base = e * (map_dim + 1)

    expected = real(e + 1, 8)
    call check(abs(fetched(base + 1) - expected) < tol, base, &
      "mixed op_arg_idx direct failed")

    do d = 0, map_dim - 1
      expected = real(f_ppedge(e * map_dim + d + 1) + 1, 8)
      call check(abs(fetched(base + d + 2) - expected) < tol, base + d + 1, &
        "mixed op_arg_idx indirect failed")
    end do
  end do
  write(*,*) "mixed direct and indirect idx passed [rank", my_rank, "]"

  deallocate(fetched)

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

end program idx_tests_fortran
