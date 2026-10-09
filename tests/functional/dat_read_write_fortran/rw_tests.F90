! Indirect read-write loops, which need colouring: edges update their nodes
! through read-write and increment arguments, so neighbouring edges and
! edges sharing a hub conflict.  Not intended to be used with OP_NO_REALLOC.

program rw_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use rw_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: g_node = 1000
  integer(4), parameter :: nhub = 7
  integer(4), parameter :: far = 5
  ! Enough passes for each loop to reach its JIT kernel.
  integer(4), parameter :: npass = 12

  type(op_set) :: nodes, edges
  type(op_map) :: m_e2n, m_e2f, m_e2h
  type(op_dat) :: pe_r
  type(op_dat) :: pn_init_i, pn_init_r
  type(op_dat) :: pn_count, pn_a, pn_b, pn_d
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  integer(4) :: g_nedge
  integer(4) :: nnode, nedge
  integer(4) :: node_size_inc_halo, edge_size_inc_halo, nexec
  integer(4) :: my_rank, comm_size
  integer(4) :: i, e, pass
  integer(4) :: n0, n1, nh

  integer(c_int), pointer, dimension(:)   :: f_m_e2n, f_m_e2h
  integer(c_int), pointer, dimension(:)   :: f_init_i
  real(c_double), pointer, dimension(:)   :: f_init_r, f_e_r

  integer(4), dimension(:), allocatable, target :: e2n, e2f, e2h, n_init_i
  real(8), dimension(:), allocatable, target :: e_r, n_init_r, n_a

  integer(4), dimension(:), allocatable :: fetched_i, expected_i
  real(8), dimension(:), allocatable :: fetched, expected, edge_sum, hub_count

  real(8) :: total, total_expected, total_local
  integer(4) :: use_d

  call op_init_base(0, 0)
  call op_profile_start("FortranReadWriteTests")

  call get_rank_and_size(my_rank, comm_size)

  g_nedge = g_node - 1

  call generate_mesh(comm_size, my_rank, nnode, nedge, e2n, e2f, e2h, e_r, &
    n_init_i, n_init_r, n_a)

  call op_decl_set(nnode, nodes, "nodes")
  call op_decl_set(nedge, edges, "edges")

  call op_decl_map(edges, nodes, 2, e2n, m_e2n, "edge_to_nodes")
  call op_decl_map(edges, nodes, 1, e2f, m_e2f, "edge_to_far")
  call op_decl_map(edges, nodes, 1, e2h, m_e2h, "edge_to_hub")

  call op_decl_dat(edges, 1, "real(8)", e_r, pe_r, "pe_r")

  ! Pristine copies of the initial node values, never written.
  call op_decl_dat(nodes, 1, "integer(4)", n_init_i, pn_init_i, "pn_init_i")
  call op_decl_dat(nodes, 1, "real(8)", n_init_r, pn_init_r, "pn_init_r")

  call op_decl_dat(nodes, 1, "integer(4)", n_init_i, pn_count, "pn_count")
  call op_decl_dat(nodes, 2, "real(8)", n_a, pn_a, "pn_a")
  call op_decl_dat(nodes, 1, "real(8)", n_init_r, pn_b, "pn_b")
  call op_decl_dat(nodes, 1, "real(8)", n_init_r, pn_d, "pn_d")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)

  node_size_inc_halo = nodes%setPtr%size + nodes%setPtr%exec_size + nodes%setPtr%nonexec_size
  edge_size_inc_halo = edges%setPtr%size + edges%setPtr%exec_size + edges%setPtr%nonexec_size
  nexec = edges%setPtr%size + edges%setPtr%exec_size

  call c_f_pointer(m_e2n%mapPtr%map, f_m_e2n, [2 * nexec])
  call c_f_pointer(m_e2h%mapPtr%map, f_m_e2h, [nexec])
  call c_f_pointer(pn_init_i%dataPtr%dat, f_init_i, [node_size_inc_halo])
  call c_f_pointer(pn_init_r%dataPtr%dat, f_init_r, [node_size_inc_halo])
  call c_f_pointer(pe_r%dataPtr%dat, f_e_r, [edge_size_inc_halo])

  ! Each owned node's contributions, from every edge this rank executes.
  allocate(expected_i(node_size_inc_halo))
  allocate(edge_sum(node_size_inc_halo))
  allocate(hub_count(node_size_inc_halo))
  expected_i = 0
  edge_sum = 0.0_8
  hub_count = 0.0_8
  do e = 0, nexec - 1
    n0 = f_m_e2n(2 * e + 1)
    n1 = f_m_e2n(2 * e + 2)
    nh = f_m_e2h(e + 1)

    expected_i(n0 + 1) = expected_i(n0 + 1) + 1
    expected_i(n1 + 1) = expected_i(n1 + 1) + 1
    expected_i(nh + 1) = expected_i(nh + 1) + 1
    edge_sum(n0 + 1) = edge_sum(n0 + 1) + f_e_r(e + 1)
    edge_sum(n1 + 1) = edge_sum(n1 + 1) + f_e_r(e + 1)
    hub_count(nh + 1) = hub_count(nh + 1) + 1.0_8
  end do

  total_local = 0.0_8
  do e = 0, edges%setPtr%size - 1
    total_local = total_local + f_e_r(e + 1)
  end do
  call global_sum(total_local, total_expected)

  ! --- Indirect integer RW through two maps ---
  do pass = 1, npass
    call op_par_loop_3(rw_count, edges, &
      op_arg_dat(pn_count, 1, m_e2n, 1, "integer(4)", OP_RW), &
      op_arg_dat(pn_count, 2, m_e2n, 1, "integer(4)", OP_RW), &
      op_arg_dat(pn_count, 1, m_e2h, 1, "integer(4)", OP_RW))
  end do

  allocate(fetched_i(node_size_inc_halo))
  call op_fetch_data(pn_count, fetched_i)

  do i = 1, nodes%setPtr%size
    call check(fetched_i(i) == f_init_i(i) + npass * expected_i(i), i, my_rank, &
      "rw_count failed")
  end do
  write(*,*) "rw_count passed [rank", my_rank, "]"

  ! --- Mixed RW, INC and READ, an optional INC and a global reduction ---
  ! The last pass drops the optional argument, so the loop changes plan.
  do pass = 1, npass
    use_d = merge(1, 0, pass < npass)
    total = 0.0_8
    call op_par_loop_9(rw_mixed, edges, &
      op_arg_dat(pn_a, 1, m_e2n, 2, "real(8)", OP_RW), &
      op_arg_dat(pn_a, 2, m_e2n, 2, "real(8)", OP_RW), &
      op_arg_dat(pn_a, 1, m_e2f, 2, "real(8)", OP_READ), &
      op_arg_dat(pn_b, 1, m_e2n, 1, "real(8)", OP_INC), &
      op_arg_dat(pn_b, 2, m_e2n, 1, "real(8)", OP_INC), &
      op_opt_arg_dat(use_d == 1, pn_d, 1, m_e2h, 1, "real(8)", OP_INC), &
      op_arg_dat(pe_r, -1, OP_ID, 1, "real(8)", OP_READ), &
      op_arg_gbl(use_d, 1, "integer(4)", OP_READ), &
      op_arg_gbl(total, 1, "real(8)", OP_INC))

    call check(abs(total - total_expected) < tol, pass, my_rank, &
      "rw_mixed global reduction failed")
  end do

  allocate(fetched(2 * node_size_inc_halo))
  allocate(expected(2 * node_size_inc_halo))

  call op_fetch_data(pn_a, fetched)
  do i = 1, nodes%setPtr%size
    expected(2 * i - 1) = f_init_r(i) + npass * edge_sum(i)
    expected(2 * i) = 2.0_8 * (f_init_r(i) + npass * edge_sum(i))
    call check(abs(fetched(2 * i - 1) - expected(2 * i - 1)) < tol, i, my_rank, &
      "rw_mixed read-write failed")
    call check(abs(fetched(2 * i) - expected(2 * i)) < tol, i, my_rank, &
      "rw_mixed read-write failed")
  end do

  call op_fetch_data(pn_b, fetched)
  do i = 1, nodes%setPtr%size
    call check(abs(fetched(i) - (f_init_r(i) + npass * edge_sum(i))) < tol, &
      i, my_rank, "rw_mixed increment failed")
  end do

  call op_fetch_data(pn_d, fetched)
  do i = 1, nodes%setPtr%size
    call check(abs(fetched(i) - (f_init_r(i) + (npass - 1) * hub_count(i))) < tol, &
      i, my_rank, "rw_mixed optional increment failed")
  end do
  write(*,*) "rw_mixed passed [rank", my_rank, "]"

  deallocate(fetched_i, expected_i, fetched, expected, edge_sum, hub_count)

  call op_profile_end()

  if (op_is_root() == 1) print *
  call op_profile_output()

  call op_exit()

contains ! ---------------------------------------------------------------------------------------------------

  subroutine global_sum(local, total)
    real(8), intent(in)  :: local
    real(8), intent(out) :: total
#ifdef USE_MPI
    integer(4) :: ierr

    call mpi_allreduce(local, total, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
      MPI_COMM_WORLD, ierr)
#else
    total = local
#endif
  end subroutine global_sum

  subroutine check(cond, idx, rank, msg)
    logical, intent(in) :: cond
    integer, intent(in) :: idx
    integer, intent(in) :: rank
    character(len=*), intent(in) :: msg

    if (.not. cond) then
      write(*,*) "ERROR:", trim(msg), " at idx:", idx, "rank:", rank
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

  ! A chain of edges, each also reaching the node a few places on and one of
  ! a few hub nodes.  Values are small integers, so every sum is exact.
  subroutine generate_mesh(mpi_comm_size, mpi_rank, nnode, nedge, e2n, e2f, &
    e2h, e_r, n_init_i, n_init_r, n_a)

    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: mpi_rank
    integer(4), intent(out) :: nnode
    integer(4), intent(out) :: nedge

    integer(4), dimension(:), allocatable, target, intent(out) :: e2n, e2f, e2h
    integer(4), dimension(:), allocatable, target, intent(out) :: n_init_i
    real(8), dimension(:), allocatable, target, intent(out) :: e_r, n_init_r, n_a

    integer(4), dimension(:), allocatable :: g_e2n, g_e2f, g_e2h, g_n_init_i
    real(8), dimension(:), allocatable :: g_e_r, g_n_init_r, g_n_a

    integer :: i, g_edges, g_nodes

    nnode = compute_local_size(g_node, mpi_comm_size, mpi_rank)
    nedge = compute_local_size(g_nedge, mpi_comm_size, mpi_rank)

    allocate(e2n(2 * nedge), e2f(nedge), e2h(nedge), e_r(nedge))
    allocate(n_init_i(nnode), n_init_r(nnode), n_a(2 * nnode))

    g_edges = 1
    g_nodes = 1
    if (mpi_rank == 0) then
      g_edges = g_nedge
      g_nodes = g_node
    end if
#ifndef USE_MPI
    g_edges = g_nedge
    g_nodes = g_node
#endif

    allocate(g_e2n(2 * g_edges), g_e2f(g_edges), g_e2h(g_edges), g_e_r(g_edges))
    allocate(g_n_init_i(g_nodes), g_n_init_r(g_nodes), g_n_a(2 * g_nodes))

    if (mpi_rank == 0) then
      do i = 0, g_nedge - 1
        g_e2n(2 * i + 1) = i
        g_e2n(2 * i + 2) = i + 1
        g_e2f(i + 1) = min(i + far, g_node - 1)
        g_e2h(i + 1) = mod(i, nhub)
        g_e_r(i + 1) = real(mod(i, 11) + 1, 8)
      end do

      do i = 0, g_node - 1
        g_n_init_i(i + 1) = mod(3 * i, 17)
        g_n_init_r(i + 1) = real(mod(5 * i, 13), 8) + 0.5_8
        g_n_a(2 * i + 1) = g_n_init_r(i + 1)
        g_n_a(2 * i + 2) = 2.0_8 * g_n_init_r(i + 1)
      end do
    end if

    call scatter_array_int(g_e2n, e2n, mpi_comm_size, g_nedge, nedge, 2)
    call scatter_array_int(g_e2f, e2f, mpi_comm_size, g_nedge, nedge, 1)
    call scatter_array_int(g_e2h, e2h, mpi_comm_size, g_nedge, nedge, 1)
    call scatter_array_real8(g_e_r, e_r, mpi_comm_size, g_nedge, nedge, 1)
    call scatter_array_int(g_n_init_i, n_init_i, mpi_comm_size, g_node, nnode, 1)
    call scatter_array_real8(g_n_init_r, n_init_r, mpi_comm_size, g_node, nnode, 1)
    call scatter_array_real8(g_n_a, n_a, mpi_comm_size, g_node, nnode, 2)

    deallocate(g_e2n, g_e2f, g_e2h, g_e_r, g_n_init_i, g_n_init_r, g_n_a)
  end subroutine generate_mesh

  subroutine scatter_counts(mpi_comm_size, g_size, dim, sendcnts, displs)
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: g_size
    integer, intent(in) :: dim
    integer, dimension(:), intent(out) :: sendcnts
    integer, dimension(:), intent(out) :: displs
    integer :: i, disp

    disp = 0
    do i = 1, mpi_comm_size
      sendcnts(i) = dim * compute_local_size(g_size, mpi_comm_size, i - 1)
      displs(i) = disp
      disp = disp + sendcnts(i)
    end do
  end subroutine scatter_counts

  subroutine scatter_array_int(g_array, l_array, mpi_comm_size, g_size, l_size, dim)
    integer(4), dimension(:), intent(in) :: g_array
    integer(4), dimension(:), intent(out) :: l_array
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: g_size
    integer, intent(in) :: l_size
    integer, intent(in) :: dim

#ifdef USE_MPI
    integer :: sendcnts(mpi_comm_size), displs(mpi_comm_size), ierr

    call scatter_counts(mpi_comm_size, g_size, dim, sendcnts, displs)
    call MPI_Scatterv(g_array, sendcnts, displs, MPI_INTEGER, &
      l_array, l_size * dim, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
#else
    l_array = g_array
#endif
  end subroutine scatter_array_int

  subroutine scatter_array_real8(g_array, l_array, mpi_comm_size, g_size, l_size, dim)
    real(8), dimension(:), intent(in) :: g_array
    real(8), dimension(:), intent(out) :: l_array
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: g_size
    integer, intent(in) :: l_size
    integer, intent(in) :: dim

#ifdef USE_MPI
    integer :: sendcnts(mpi_comm_size), displs(mpi_comm_size), ierr

    call scatter_counts(mpi_comm_size, g_size, dim, sendcnts, displs)
    call MPI_Scatterv(g_array, sendcnts, displs, MPI_DOUBLE_PRECISION, &
      l_array, l_size * dim, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
#else
    l_array = g_array
#endif
  end subroutine scatter_array_real8

end program rw_tests_fortran
