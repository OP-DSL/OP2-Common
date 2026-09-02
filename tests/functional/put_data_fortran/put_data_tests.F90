! Not intended to be used with OP_NO_REALLOC flag

program put_data_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use put_data_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: nn = 48

  type(op_set) :: nodes, edges
  type(op_map) :: m_e2n
  type(op_dat) :: pn_dat1, pn_dat3, pe_dat4
  type(op_dat) :: pn_out1, pn_out3, pe_out4
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  integer(4), dimension(:), allocatable, target :: e2n
  real(8), dimension(:), allocatable, target :: n_init1, n_init3, e_init4
  real(8), dimension(:), allocatable, target :: n_out1, n_out3, e_out4
  real(8), dimension(:), allocatable, target :: put_n1, put_n3, put_e4
  real(8), dimension(:), allocatable :: fetched

  integer(4) :: g_node, g_nedge
  integer(4) :: orig_nnode, orig_nedge
  integer(4) :: node_start, edge_start
  integer(4) :: my_rank, comm_size
  integer(4) :: i, d, g

  call op_init_base(0, 0)
  call op_profile_start("FortranPutDataTests")

  call get_rank_and_size(my_rank, comm_size)

  g_node = nn
  g_nedge = nn - 1
  orig_nnode = compute_local_size(g_node, comm_size, my_rank)
  orig_nedge = compute_local_size(g_nedge, comm_size, my_rank)
  node_start = get_local_start(g_node, comm_size, my_rank)
  edge_start = get_local_start(g_nedge, comm_size, my_rank)

  write(*,*) "Global number of nodes, edges =", g_node, g_nedge
  write(*,*) "Number of nodes, edges on process", my_rank, "=", orig_nnode, orig_nedge

  allocate(e2n(2 * orig_nedge))
  do i = 0, orig_nedge - 1
    g = edge_start + i
    e2n(2 * i + 1) = g
    e2n(2 * i + 2) = g + 1
    if (e2n(2 * i + 2) >= g_node) e2n(2 * i + 2) = g_node - 1
  end do

  allocate(n_init1(orig_nnode))
  allocate(n_init3(orig_nnode * 3))
  allocate(e_init4(orig_nedge * 4))
  allocate(n_out1(orig_nnode))
  allocate(n_out3(orig_nnode * 3))
  allocate(e_out4(orig_nedge * 4))
  n_init1 = -1.0d0
  n_init3 = -1.0d0
  e_init4 = -1.0d0
  n_out1 = 0.0d0
  n_out3 = 0.0d0
  e_out4 = 0.0d0

  call op_decl_set(orig_nnode, nodes, "nodes")
  call op_decl_set(orig_nedge, edges, "edges")
  call op_decl_map(edges, nodes, 2, e2n, m_e2n, "edge_to_nodes")

  call op_decl_dat(nodes, 1, "real(8)", n_init1, pn_dat1, "pn_dat1")
  call op_decl_dat(nodes, 3, "real(8)", n_init3, pn_dat3, "pn_dat3")
  call op_decl_dat(edges, 4, "real(8)", e_init4, pe_dat4, "pe_dat4")
  call op_decl_dat(nodes, 1, "real(8)", n_out1, pn_out1, "pn_out1")
  call op_decl_dat(nodes, 3, "real(8)", n_out3, pn_out3, "pn_out3")
  call op_decl_dat(edges, 4, "real(8)", e_out4, pe_out4, "pe_out4")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
#ifdef USE_MPI
  call op_partition("RANDOM", "", nodes, dummy_map, dummy_dat)
#else
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)
#endif

  allocate(put_n1(orig_nnode))
  allocate(put_n3(orig_nnode * 3))
  allocate(put_e4(orig_nedge * 4))

  do i = 0, orig_nnode - 1
    g = node_start + i
    put_n1(i + 1) = real(g + 1, 8) * 17.0d0
    do d = 0, 2
      put_n3(i * 3 + d + 1) = real(g + 1, 8) * 17.0d0 + 0.125d0 * d
    end do
  end do
  do i = 0, orig_nedge - 1
    g = edge_start + i
    do d = 0, 3
      put_e4(i * 4 + d + 1) = real(g + 1, 8) * 3.0d0 + 1000.5d0 * d
    end do
  end do

  call op_mpi_put_data(pn_dat1, put_n1, orig_nnode)
  call op_mpi_put_data(pn_dat3, put_n3, orig_nnode)
  call op_mpi_put_data(pe_dat4, put_e4, orig_nedge)

  allocate(fetched(orig_nnode))
  call op_fetch_data(pn_dat1, fetched)
  do i = 1, orig_nnode
    call check(abs(fetched(i) - put_n1(i)) < tol, i, my_rank, &
      "put/fetch dim=1 nodes failed")
  end do
  write(*,*) "put/fetch dim=1 nodes passed [rank", my_rank, "]"
  deallocate(fetched)

  allocate(fetched(orig_nnode * 3))
  call op_fetch_data(pn_dat3, fetched)
  do i = 1, orig_nnode * 3
    call check(abs(fetched(i) - put_n3(i)) < tol, i, my_rank, &
      "put/fetch dim=3 nodes failed")
  end do
  write(*,*) "put/fetch dim=3 nodes passed [rank", my_rank, "]"
  deallocate(fetched)

  allocate(fetched(orig_nedge * 4))
  call op_fetch_data(pe_dat4, fetched)
  do i = 1, orig_nedge * 4
    call check(abs(fetched(i) - put_e4(i)) < tol, i, my_rank, &
      "put/fetch dim=4 edges failed")
  end do
  write(*,*) "put/fetch dim=4 edges passed [rank", my_rank, "]"
  deallocate(fetched)

  call op_par_loop_2(copy1, nodes, &
    op_arg_dat(pn_out1, -1, OP_ID, 1, "real(8)", OP_WRITE), &
    op_arg_dat(pn_dat1, -1, OP_ID, 1, "real(8)", OP_READ))

  allocate(fetched(orig_nnode))
  call op_fetch_data(pn_out1, fetched)
  do i = 1, orig_nnode
    call check(abs(fetched(i) - put_n1(i)) < tol, i, my_rank, &
      "kernel after put dim=1 nodes failed")
  end do
  write(*,*) "kernel after put dim=1 nodes passed [rank", my_rank, "]"
  deallocate(fetched)

  call op_par_loop_2(copy3, nodes, &
    op_arg_dat(pn_out3, -1, OP_ID, 3, "real(8)", OP_WRITE), &
    op_arg_dat(pn_dat3, -1, OP_ID, 3, "real(8)", OP_READ))

  allocate(fetched(orig_nnode * 3))
  call op_fetch_data(pn_out3, fetched)
  do i = 1, orig_nnode * 3
    call check(abs(fetched(i) - put_n3(i)) < tol, i, my_rank, &
      "kernel after put dim=3 nodes failed")
  end do
  write(*,*) "kernel after put dim=3 nodes passed [rank", my_rank, "]"
  deallocate(fetched)

  call op_par_loop_2(copy4, edges, &
    op_arg_dat(pe_out4, -1, OP_ID, 4, "real(8)", OP_WRITE), &
    op_arg_dat(pe_dat4, -1, OP_ID, 4, "real(8)", OP_READ))

  allocate(fetched(orig_nedge * 4))
  call op_fetch_data(pe_out4, fetched)
  do i = 1, orig_nedge * 4
    call check(abs(fetched(i) - put_e4(i)) < tol, i, my_rank, &
      "kernel after put dim=4 edges failed")
  end do
  write(*,*) "kernel after put dim=4 edges passed [rank", my_rank, "]"
  deallocate(fetched)

  call op_profile_end()

  if (op_is_root() == 1) print *
  call op_profile_output()

  call op_exit()

contains

  subroutine check(cond, idx, rank, msg)
    logical, intent(in) :: cond
    integer, intent(in) :: idx, rank
    character(len=*), intent(in) :: msg

    if (.not. cond) then
      write(*,*) "ERROR:", trim(msg), " at idx:", idx, " rank:", rank
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

end program put_data_tests_fortran
