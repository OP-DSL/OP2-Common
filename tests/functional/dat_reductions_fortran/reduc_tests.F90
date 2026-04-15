! Not intended to be used with OP_NO_REALLOC flag

program reduc_tests_fortran
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use reduc_kernels

  use, intrinsic :: iso_c_binding
#ifdef USE_MPI
  use mpi
#endif

  implicit none

  real(8), parameter :: tol = 1.0d-9
  integer(4), parameter :: nn = 12

  type(op_set) :: nodes, edges
  type(op_map) :: m_e2n
  type(op_dat) :: pe_dat1, pe_dat1_u, pe_dat4, pe_dat4_u
  type(op_dat) :: pn_dat1, pn_dat1_u, pn_dat3, pn_dat3_u
  type(op_set) :: dummy_set
  type(op_map) :: dummy_map
  type(op_dat) :: dummy_dat

  integer(4) :: g_node, g_nedge
  integer(4) :: nnode, nedge
  integer(4) :: node_size_inc_halo, edge_size_inc_halo
  integer(4) :: my_rank, comm_size
  integer(4) :: i, d, e
  integer(4) :: n0, n1
  
  integer(c_int), pointer, dimension(:)   :: f_m_e2n
  real(c_float), pointer, dimension(:, :) :: f_e_dat1, f_e_dat4, f_n_dat1, f_n_dat3

  integer(4), dimension(:), allocatable, target :: e2n
  real(4), dimension(:), allocatable, target :: e_dat1
  real(4), dimension(:), allocatable, target :: e_dat4
  real(4), dimension(:), allocatable, target :: n_dat1
  real(4), dimension(:), allocatable, target :: n_dat3

  real(4), dimension(:), allocatable :: fetched
  real(4), dimension(:), allocatable :: expected

  call op_init_base(0, 0)

  call get_rank_and_size(my_rank, comm_size)

  g_node = nn * nn
  g_nedge = g_node - 1

  call generate_1d_umesh(g_node, g_nedge, comm_size, my_rank, nnode, nedge, &
    e2n, e_dat1, e_dat4, n_dat1, n_dat3)

  call op_decl_set(nnode, nodes, "nodes")
  call op_decl_set(nedge, edges, "edges")

  call op_decl_map(edges, nodes, 2, e2n, m_e2n, "edge_to_nodes")

  call op_decl_dat(edges, 1, "real(4)", e_dat1, pe_dat1, "pe_dat1")
  call op_decl_dat(edges, 1, "real(4)", e_dat1, pe_dat1_u, "pe_dat1_u")
  call op_decl_dat(edges, 4, "real(4)", e_dat4, pe_dat4, "pe_dat4")
  call op_decl_dat(edges, 4, "real(4)", e_dat4, pe_dat4_u, "pe_dat4_u")

  call op_decl_dat(nodes, 1, "real(4)", n_dat1, pn_dat1, "pn_dat1")
  call op_decl_dat(nodes, 1, "real(4)", n_dat1, pn_dat1_u, "pn_dat1_u")
  call op_decl_dat(nodes, 3, "real(4)", n_dat3, pn_dat3, "pn_dat3")
  call op_decl_dat(nodes, 3, "real(4)", n_dat3, pn_dat3_u, "pn_dat3_u")

  call nullify_dummy(dummy_set, dummy_map, dummy_dat)
  call op_partition("", "", dummy_set, dummy_map, dummy_dat)

  node_size_inc_halo = nodes%setPtr%size + nodes%setPtr%exec_size + nodes%setPtr%nonexec_size
  edge_size_inc_halo = edges%setPtr%size + edges%setPtr%exec_size + edges%setPtr%nonexec_size
  
  ! Get dat data pointers for evaluations
  call c_f_pointer(m_e2n%mapPtr%map, f_m_e2n, [edges%setPtr%size + edges%setPtr%exec_size])
  call c_f_pointer(pn_dat1%dataPtr%dat, f_n_dat1, (/1, node_size_inc_halo/))
  call c_f_pointer(pn_dat3%dataPtr%dat, f_n_dat3, (/3, node_size_inc_halo/))
  call c_f_pointer(pe_dat1%dataPtr%dat, f_e_dat1, (/1, edge_size_inc_halo/))
  call c_f_pointer(pe_dat4%dataPtr%dat, f_e_dat4, (/4, edge_size_inc_halo/))

  ! --- Indirect Dat INC DIM=1 ---
  call op_par_loop_3(indirect_dat1_inc, edges, &
    op_arg_dat(pn_dat1_u, 1, m_e2n, 1, "real(4)", OP_INC), &
    op_arg_dat(pn_dat1_u, 2, m_e2n, 1, "real(4)", OP_INC), &
    op_arg_dat(pe_dat1, -1, OP_ID, 1, "real(4)", OP_READ))

  allocate(fetched(node_size_inc_halo))
  call op_fetch_data(pn_dat1_u, fetched)

  allocate(expected(node_size_inc_halo))
  expected = 0.0
  do i = 1, node_size_inc_halo
    expected(i) = f_n_dat1(1, i)
  end do

  do e = 0, edges%setPtr%size + edges%setPtr%exec_size - 1
    n0 = f_m_e2n(2 * e + 1)
    n1 = f_m_e2n(2 * e + 2)

    expected(n0 + 1) = expected(n0 + 1) + f_e_dat1(1, e + 1)
    expected(n1 + 1) = expected(n1 + 1) + f_e_dat1(1, e + 1)
  end do

  do i = 1, nodes%setPtr%size
    call check(abs(real(fetched(i)) - real(expected(i))) < tol, i, my_rank, &
      "indirect_dat1_inc failed")
  end do
  write(*,*) "indirect_dat1_inc passed [rank", my_rank, "]"

  deallocate(fetched)
  deallocate(expected)

  ! --- Indirect Dat INC DIM=3 ---
  call op_par_loop_3(indirect_dat3_inc, edges, &
    op_arg_dat(pn_dat3_u, 1, m_e2n, 3, "real(4)", OP_INC), &
    op_arg_dat(pn_dat3_u, 2, m_e2n, 3, "real(4)", OP_INC), &
    op_arg_dat(pe_dat4, -1, OP_ID, 4, "real(4)", OP_READ))

  allocate(fetched(node_size_inc_halo * 3))
  call op_fetch_data(pn_dat3_u, fetched)

  allocate(expected(node_size_inc_halo * 3))
  expected = 0.0
  do i = 0, node_size_inc_halo - 1
    do d = 1, 3
      expected(i * 3 + d) = f_n_dat3(d, i + 1)
    end do
  end do

 do e = 0, edges%setPtr%size + edges%setPtr%exec_size - 1
   n0 = f_m_e2n(2 * e + 1)
   n1 = f_m_e2n(2 * e + 2)

   do d = 1, 3
     expected(3 * n0 + d) = expected(3 * n0 + d) + f_e_dat4(d, e + 1)
     expected(3 * n1 + d) = expected(3 * n1 + d) + f_e_dat4(d, e + 1)
   end do
 end do

 do i = 0, nodes%setPtr%size - 1
   do d = 1, 3
     call check(abs(real(fetched(i * 3 + d)) - real(expected(i * 3 + d))) < tol, &
       i * 3 + (d - 1), my_rank, "indirect_dat3_inc failed")
   end do
 end do
 write(*,*) "indirect_dat3_inc passed [rank", my_rank, "]"

 deallocate(fetched)
 deallocate(expected)

 ! --- Direct Dat INC DIM=1 ---
 call op_par_loop_2(direct_dat1_inc, edges, &
   op_arg_dat(pe_dat1, -1, OP_ID, 1, "real(4)", OP_READ), &
   op_arg_dat(pe_dat1_u, -1, OP_ID, 1, "real(4)", OP_INC))

 allocate(fetched(edge_size_inc_halo))
 call op_fetch_data(pe_dat1_u, fetched)

 do i = 1, edges%setPtr%size
   call check(abs(real(fetched(i)) - real(2.0 * f_e_dat1(1, i) + 3.25)) < tol, &
     i, my_rank, "direct_dat1_inc failed")
 end do
 write(*,*) "direct_dat1_inc passed [rank", my_rank, "]"

 deallocate(fetched)

 ! --- Direct Dat INC DIM=4 ---
 call op_par_loop_2(direct_dat4_inc, edges, &
   op_arg_dat(pe_dat4, -1, OP_ID, 4, "real(4)", OP_READ), &
   op_arg_dat(pe_dat4_u, -1, OP_ID, 4, "real(4)", OP_INC))

 allocate(fetched(edge_size_inc_halo * 4))
 call op_fetch_data(pe_dat4_u, fetched)

 do i = 0, edges%setPtr%size - 1
   do d = 1, 4
     call check(abs(real(fetched(i * 4 + d)) - real(2.0 * f_e_dat4(d, i + 1) + 1.325 * real(d - 1))) < tol, &
       i * 4 + (d - 1), my_rank, "direct_dat4_inc failed")
   end do
 end do
 write(*,*) "direct_dat4_inc passed [rank", my_rank, "]"

 deallocate(fetched)

  call op_exit()

contains ! ---------------------------------------------------------------------------------------------------

  ! --- Utility functions ---
  subroutine check(cond, idx, rank, msg)
    logical, intent(in) :: cond
    integer, intent(in) :: idx
    integer, intent(in) :: rank
    character(len=*), intent(in) :: msg

    if (.not. cond) then
      write(*,*) "ERROR:", trim(msg), "at idx:", idx, "rank:", rank
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

  subroutine generate_1d_umesh(g_node, g_nedge, mpi_comm_size, mpi_rank, nnode, nedge, &
    e2n, e_dat1, e_dat4, n_dat1, n_dat3)

    integer(4), intent(in) :: g_node
    integer(4), intent(in) :: g_nedge
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: mpi_rank
    integer(4), intent(out) :: nnode
    integer(4), intent(out) :: nedge

    integer(4), dimension(:), allocatable, target, intent(out) :: e2n
    real(4), dimension(:), allocatable, target, intent(out) :: e_dat1
    real(4), dimension(:), allocatable, target, intent(out) :: e_dat4
    real(4), dimension(:), allocatable, target, intent(out) :: n_dat1
    real(4), dimension(:), allocatable, target, intent(out) :: n_dat3

    integer(4), dimension(:), allocatable :: g_e2n
    real(4), dimension(:), allocatable :: g_e_dat1
    real(4), dimension(:), allocatable :: g_e_dat4
    real(4), dimension(:), allocatable :: g_n_dat1
    real(4), dimension(:), allocatable :: g_n_dat3

    integer :: i, d

    write(*,*) "Global number of nodes, edges =", g_node, g_nedge

    nnode = compute_local_size(g_node, mpi_comm_size, mpi_rank)
    nedge = compute_local_size(g_nedge, mpi_comm_size, mpi_rank)

    write(*,*) "Number of nodes, edges on process", mpi_rank, "=", nnode, nedge

    allocate(e2n(2 * nedge))
    allocate(e_dat1(nedge))
    allocate(e_dat4(nedge * 4))
    allocate(n_dat1(nnode))
    allocate(n_dat3(nnode * 3))

#ifdef USE_MPI
    if (mpi_rank == 0) then
      allocate(g_e2n(2 * g_nedge))
      allocate(g_e_dat1(g_nedge))
      allocate(g_e_dat4(g_nedge * 4))
      allocate(g_n_dat1(g_node))
      allocate(g_n_dat3(g_node * 3))
    else
      allocate(g_e2n(1))
      allocate(g_e_dat1(1))
      allocate(g_e_dat4(1))
      allocate(g_n_dat1(1))
      allocate(g_n_dat3(1))
    end if
#else
    allocate(g_e2n(2 * g_nedge))
    allocate(g_e_dat1(g_nedge))
    allocate(g_e_dat4(g_nedge * 4))
    allocate(g_n_dat1(g_node))
    allocate(g_n_dat3(g_node * 3))
#endif

    if (mpi_rank == 0) then
      do i = 0, g_nedge - 1
        g_e2n(2 * i + 1) = i
        g_e2n(2 * i + 2) = i + 1
        g_e_dat1(i + 1) = real(i, 4) * 7.0
        do d = 0, 3
          g_e_dat4(i * 4 + d + 1) = real(i, 4) * 3.0 + 1000.5 * real(d, 4)
        end do
      end do

      do i = 0, g_node - 1
        g_n_dat1(i + 1) = real(i, 4) * 13.0
        do d = 0, 2
          g_n_dat3(i * 3 + d + 1) = real(i, 4) * 2.3 + 300.5 * real(d, 4)
        end do
      end do
    end if

    call scatter_array_int(g_e2n, e2n, mpi_comm_size, g_nedge, nedge, 2)
    call scatter_array_real4(g_e_dat1, e_dat1, mpi_comm_size, g_nedge, nedge, 1)
    call scatter_array_real4(g_e_dat4, e_dat4, mpi_comm_size, g_nedge, nedge, 4)
    call scatter_array_real4(g_n_dat1, n_dat1, mpi_comm_size, g_node, nnode, 1)
    call scatter_array_real4(g_n_dat3, n_dat3, mpi_comm_size, g_node, nnode, 3)

    deallocate(g_e2n)
    deallocate(g_e_dat1)
    deallocate(g_e_dat4)
    deallocate(g_n_dat1)
    deallocate(g_n_dat3)
  end subroutine generate_1d_umesh

  subroutine scatter_array_int(g_array, l_array, mpi_comm_size, g_size, l_size, dim)
    integer(4), dimension(:), allocatable, intent(in) :: g_array
    integer(4), dimension(:), intent(out) :: l_array
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: g_size
    integer, intent(in) :: l_size
    integer, intent(in) :: dim

#ifdef USE_MPI
    integer, allocatable :: sendcnts(:)
    integer, allocatable :: displs(:)
    integer :: i, disp, ierr

    allocate(sendcnts(mpi_comm_size))
    allocate(displs(mpi_comm_size))

    disp = 0
    do i = 0, mpi_comm_size - 1
      sendcnts(i + 1) = dim * compute_local_size(g_size, mpi_comm_size, i)
    end do

    do i = 1, mpi_comm_size
      displs(i) = disp
      disp = disp + sendcnts(i)
    end do

    call MPI_Scatterv(g_array, sendcnts, displs, MPI_INTEGER, &
      l_array, l_size * dim, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    deallocate(sendcnts)
    deallocate(displs)
#else
    l_array = g_array
#endif
  end subroutine scatter_array_int

  subroutine scatter_array_real4(g_array, l_array, mpi_comm_size, g_size, l_size, dim)
    real(4), dimension(:), allocatable, intent(in) :: g_array
    real(4), dimension(:), intent(out) :: l_array
    integer, intent(in) :: mpi_comm_size
    integer, intent(in) :: g_size
    integer, intent(in) :: l_size
    integer, intent(in) :: dim

#ifdef USE_MPI
    integer, allocatable :: sendcnts(:)
    integer, allocatable :: displs(:)
    integer :: i, disp, ierr

    allocate(sendcnts(mpi_comm_size))
    allocate(displs(mpi_comm_size))

    disp = 0
    do i = 0, mpi_comm_size - 1
      sendcnts(i + 1) = dim * compute_local_size(g_size, mpi_comm_size, i)
    end do

    do i = 1, mpi_comm_size
      displs(i) = disp
      disp = disp + sendcnts(i)
    end do

    call MPI_Scatterv(g_array, sendcnts, displs, MPI_REAL, &
      l_array, l_size * dim, MPI_REAL, 0, MPI_COMM_WORLD, ierr)

    deallocate(sendcnts)
    deallocate(displs)
#else
    l_array = g_array
#endif
  end subroutine scatter_array_real4

end program reduc_tests_fortran
