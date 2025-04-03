program jac_distributed

  ! Use OP2 Fortran bindings and MPI
  use OP2_FORTRAN_DECLARATIONS
  use OP2_FORTRAN_REFERENCE
  use op2_fortran_rt_support
  use, intrinsic :: ISO_C_BINDING
  use mpi

  implicit none

  ! Define integer kind for global indices (matches C++ idx_g_t)
  integer, parameter :: idx_k = selected_int_kind(18) ! Kind for 64-bit integers
  integer(idx_k), parameter :: nn = int(2**15, kind=idx_k) ! Problem size (ensure it's idx_k)
  integer, parameter :: niter = 2                     ! Number of iterations
  real(8), parameter :: tolerance = 1.0e-12_8         ! Validation tolerance

  ! Global constants
  real(8) :: alpha

  ! MPI variables
  integer :: my_rank, comm_size, ierr

  ! Local mesh data
  integer(idx_k) :: nnode       ! Local number of nodes
  integer(idx_k) :: nedge       ! Local number of edges
  integer(idx_k) :: node_start  ! Starting global index for local nodes
  integer(idx_k) :: g_nnode     ! Global number of nodes

  ! Allocatable arrays for local data
  integer(idx_k), dimension(:), allocatable :: pp  ! Edge connectivity (stores global indices)
  real(8), dimension(:), allocatable :: A     ! Edge weights
  real(8), dimension(:), allocatable :: r     ! Node residual/RHS
  real(8), dimension(:), allocatable :: u     ! Node solution
  real(8), dimension(:), allocatable :: du    ! Node update

  ! OP2 objects
  type(op_set) :: nodes, edges
  type(op_map) :: ppedge
  type(op_dat) :: p_A, p_r, p_u, p_du

  ! Iteration variables
  integer :: iter
  real(8) :: u_sum, u_max, beta
  integer :: validation_result

  ! Temporary variables for initialization
  integer(idx_k) :: nedge_local, edge_counter
  integer(idx_k) :: local_idx, global_idx, i, j, i2, j2, neighbor_global_idx
  integer :: pass

  !--------------------------------------------------------------------------
  ! 1. Initialize MPI and OP2
  !--------------------------------------------------------------------------
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

  ! Initialize OP2 (level 2 for some diagnostics)
  call op_init(2)

  !--------------------------------------------------------------------------
  ! 2. Distributed Mesh Initialization
  !--------------------------------------------------------------------------

  ! Calculate global node count
  g_nnode = (nn - 1) * (nn - 1)
  if (my_rank == mpi_root) then
     print *, "Global number of nodes = ", g_nnode
  end if

  ! Calculate local node count and starting global index for this rank
  nnode = compute_local_size(g_nnode, int(comm_size, idx_k), int(my_rank, idx_k))
  node_start = compute_local_offset(g_nnode, int(comm_size, idx_k), int(my_rank, idx_k))

  ! Calculate local edge count (self edge + interior neighbors)
  nedge_local = 0_idx_k
  do local_idx = 1_idx_k, nnode ! Loop over local nodes
    global_idx = node_start + local_idx - 1_idx_k ! 0-based global index for calculation
    ! Recover global (i, j) from global_idx (1-based i, j in [1, nn-1])
    j = global_idx / (nn - 1) + 1_idx_k
    i = mod(global_idx, nn - 1) + 1_idx_k

    nedge_local = nedge_local + 1_idx_k ! Self edge

    ! Check 4 neighbors
    do pass = 0, 3
      i2 = i; j2 = j
      if (pass == 0) i2 = i2 - 1_idx_k
      if (pass == 1) i2 = i2 + 1_idx_k
      if (pass == 2) j2 = j2 - 1_idx_k
      if (pass == 3) j2 = j2 + 1_idx_k

      ! If neighbor is interior, count an edge
      if (i2 > 0_idx_k .and. i2 < nn .and. j2 > 0_idx_k .and. j2 < nn) then
        nedge_local = nedge_local + 1_idx_k
      end if
    end do
  end do
  nedge = nedge_local
  print *, "Process ", my_rank, ": number of local nodes, edges = ", nnode, nedge


  ! Allocate local arrays based on calculated local sizes
  allocate(pp(2 * nedge), stat=ierr); if (ierr /= 0) stop 'Allocation failed for pp'
  allocate(A(nedge), stat=ierr); if (ierr /= 0) stop 'Allocation failed for A'
  allocate(r(nnode), stat=ierr); if (ierr /= 0) stop 'Allocation failed for r'
  allocate(u(nnode), stat=ierr); if (ierr /= 0) stop 'Allocation failed for u'
  allocate(du(nnode), stat=ierr); if (ierr /= 0) stop 'Allocation failed for du'

  ! Fill local arrays (connectivity 'pp' uses global indices)
  edge_counter = 0_idx_k
  do local_idx = 1_idx_k, nnode
    global_idx = node_start + local_idx - 1_idx_k ! 0-based global index
    j = global_idx / (nn - 1) + 1_idx_k
    i = mod(global_idx, nn - 1) + 1_idx_k

    ! Initialize local node data
    r(local_idx) = 0.0_8
    u(local_idx) = 0.0_8
    du(local_idx) = 0.0_8

    ! Add self edge (using 0-based global index convention for OP2 maps)
    edge_counter = edge_counter + 1_idx_k
    pp(2 * edge_counter - 1) = global_idx ! from node
    pp(2 * edge_counter)     = global_idx ! to node
    A(edge_counter)          = -1.0_8

    ! Add edges to neighbors
    do pass = 0, 3
      i2 = i; j2 = j
      if (pass == 0) i2 = i2 - 1_idx_k
      if (pass == 1) i2 = i2 + 1_idx_k
      if (pass == 2) j2 = j2 - 1_idx_k
      if (pass == 3) j2 = j2 + 1_idx_k

      if (i2 == 0_idx_k .or. i2 == nn .or. j2 == 0_idx_k .or. j2 == nn) then
        ! Boundary neighbor: update RHS
        r(local_idx) = r(local_idx) + 0.25_8
      else
        ! Interior neighbor: add edge
        neighbor_global_idx = (i2 - 1) + (j2 - 1) * (nn - 1) ! 0-based global index
        edge_counter = edge_counter + 1_idx_k
        pp(2 * edge_counter - 1) = global_idx          ! from node
        pp(2 * edge_counter)     = neighbor_global_idx ! to node
        A(edge_counter)          = 0.25_8
      end if
    end do
  end do

  ! Check if edge counter matches expected local edge count
  if (edge_counter /= nedge) then
     print *, "Rank ", my_rank, ": Mismatch in edge count! Calculated=", nedge, " Filled=", edge_counter
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  !--------------------------------------------------------------------------
  ! 3. Declare OP2 Sets, Maps, Dats using local sizes and data
  !--------------------------------------------------------------------------
  call op_decl_set(nnode, nodes, "nodes")
  call op_decl_set(nedge, edges, "edges")

  ! Use op_decl_map_long for 64-bit integer indices in pp
  call op_decl_map_long(edges, nodes, 2, pp, ppedge, "ppedge")

  call op_decl_dat(edges, 1, "real(8)", A, p_A, "p_A")
  call op_decl_dat(nodes, 1, "real(8)", r, p_r, "p_r")
  call op_decl_dat(nodes, 1, "real(8)", u, p_u, "p_u")
  call op_decl_dat(nodes, 1, "real(8)", du, p_du, "p_du")

  ! Deallocate host arrays after declaration (OP2 holds the data)
  ! Keep 'u' for fetching results later. Keep 'pp' if needed for debugging.
  deallocate(A)
  deallocate(r)
  deallocate(du)
  ! deallocate(pp) ! Keep pp if op_decl_map_long doesn't copy immediately or for debugging

  ! Declare global constant alpha
  alpha = 1.0_8
  call op_decl_const(alpha, 1, "real(8)")

  !--------------------------------------------------------------------------
  ! 4. Partitioning (triggers data movement)
  !--------------------------------------------------------------------------
  call op_partition("PARMETIS", "KWAY", edges, ppedge, p_u) ! Or use NULL directly if allowed

  !--------------------------------------------------------------------------
  ! 5. Main Iteration Loop
  !--------------------------------------------------------------------------
  call op_timing2_enter("Main computation") ! Start timing after setup/partitioning

  beta = 1.0_8

  do iter = 1, niter
    ! Residual calculation loop (res)
    ! Arguments: A (edge, read), u (node via map idx 1, read), du (node via map idx 0, inc), beta (gbl, read)
    ! Note: C++ uses ppedge indices 0 and 1. Fortran map indices are 1-based.
    ! Map dimension 1 is pp(*,1) -> index 0 in C++ -> from node
    ! Map dimension 2 is pp(*,2) -> index 1 in C++ -> to node
    ! So, op_arg_dat(p_u, 2, ppedge, ...) reads u at the 'to' node.
    ! And op_arg_dat(p_du, 1, ppedge, ...) increments du at the 'from' node.
    call op_par_loop_4(res_kernel, edges, &
         op_arg_dat(p_A,  -1, OP_ID,    1, "real(8)", OP_READ), & ! A[edge]
         op_arg_dat(p_u,   2, ppedge,   1, "real(8)", OP_READ), & ! u[to_node]
         op_arg_dat(p_du,  1, ppedge,   1, "real(8)", OP_INC),  & ! du[from_node] += ...
         op_arg_gbl(beta, 1, "real(8)", OP_READ))

    ! Update loop (update)
    u_sum = 0.0_8
    u_max = 0.0_8
    ! Arguments: r (node, read), du (node, rw), u (node, inc), u_sum (gbl, inc), u_max (gbl, max)
    call op_par_loop_5(update_kernel, nodes, &
         op_arg_dat(p_r,  -1, OP_ID,    1, "real(8)", OP_READ), & ! r[node]
         op_arg_dat(p_du, -1, OP_ID,    1, "real(8)", OP_RW),   & ! du[node] (read and zeroed)
         op_arg_dat(p_u,  -1, OP_ID,    1, "real(8)", OP_INC),  & ! u[node] += ...
         op_arg_gbl(u_sum,              1, "real(8)", OP_INC),  &
         op_arg_gbl(u_max,              1, "real(8)", OP_MAX))

     if (my_rank == mpi_root) then
        write (*, "(4X, I0, E16.7, 4X, A, E16.7)") iter, u_max, &
            "u rms = ", sqrt(u_sum / dble(g_nnode))
     end if
  end do

  call op_timing2_finish() ! Stop timing

  !--------------------------------------------------------------------------
  ! 6. Output Timings and Fetch Results
  !--------------------------------------------------------------------------
  call op_timing2_output()

  ! Re-allocate u if it was deallocated earlier, or just use the existing one
  ! Ensure 'u' is allocated with the correct *local* size 'nnode'
  if (.not. allocated(u)) allocate(u(nnode), stat=ierr); if (ierr /= 0) stop 'Allocation failed for u (fetch)'

  ! Fetch the final solution u back into the local host array
  call op_fetch_data(p_u, u)

  !--------------------------------------------------------------------------
  ! 7. Distributed Validation
  !--------------------------------------------------------------------------
  validation_result = distributed_check_result(u, nn, node_start, nnode, tolerance, my_rank)

  !--------------------------------------------------------------------------
  ! 8. Finalize
  !--------------------------------------------------------------------------
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call op_exit()

  ! Deallocate remaining host arrays
  if (allocated(u)) deallocate(u)
  if (allocated(pp)) deallocate(pp)

  call MPI_Finalize(ierr)

  ! Exit code 0 for success (no validation failure), 1 for failure
  if (validation_result /= 0 .and. my_rank == mpi_root) then
     print *, "Exiting with status 1 due to validation failure."
     call exit(1)
  else if (my_rank == mpi_root) then
     print *, "Exiting with status 0 (success)."
  end if


contains !##################### Internal Subroutines and Functions #####################

  !--------------------------------------------------------------------------
  ! OP2 Kernels
  !--------------------------------------------------------------------------
  subroutine res_kernel(A, u, du, beta)
    ! Kernel to compute residual contribution: du[from] += beta * A[edge] * u[to]
    implicit none
    real(8), dimension(1), intent(in)  :: A     ! OP_READ from p_A (edge data)
    real(8), dimension(1), intent(in)  :: u     ! OP_READ from p_u (mapped via ppedge index 2 - 'to' node)
    real(8), dimension(1), intent(inout):: du    ! OP_INC to p_du (mapped via ppedge index 1 - 'from' node)
    real(8), dimension(1), intent(in)  :: beta  ! OP_READ from global beta

    du(1) = du(1) + beta(1) * A(1) * u(1)
  end subroutine res_kernel

  subroutine update_kernel(r, du, u, u_sum, u_max)
    ! Kernel to update solution: u += du + alpha*r; du = 0; compute global sums/max
    implicit none
    real(8), dimension(1), intent(in)    :: r      ! OP_READ from p_r (node data)
    real(8), dimension(1), intent(inout) :: du     ! OP_RW from p_du (node data)
    real(8), dimension(1), intent(inout) :: u      ! OP_INC to p_u (node data) - Note: C++ uses OP_INC
    real(8), dimension(1), intent(inout) :: u_sum  ! OP_INC to global u_sum
    real(8), dimension(1), intent(inout) :: u_max  ! OP_MAX to global u_max


    u(1) = u(1) + du(1) + alpha * r(1)
    du(1) = 0.0_8                                  ! Reset du

    u_sum(1) = u_sum(1) + u(1) ** 2
    u_max(1) = max(u_max(1), u(1))

  end subroutine update_kernel

  !--------------------------------------------------------------------------
  ! Partitioning Helper Functions
  !--------------------------------------------------------------------------
  function compute_local_size(global_size, mpi_comm_size, mpi_rank) result(local_size)
    ! Computes the local size for a given rank in a uniform distribution
    implicit none
    integer(idx_k), intent(in) :: global_size, mpi_comm_size, mpi_rank
    integer(idx_k) :: local_size
    integer(idx_k) :: base, remainder

    base = global_size / mpi_comm_size
    remainder = mod(global_size, mpi_comm_size)

    if (mpi_rank < remainder) then
      local_size = base + 1_idx_k
    else
      local_size = base
    end if
  end function compute_local_size

  function compute_local_offset(global_size, mpi_comm_size, mpi_rank) result(offset)
    ! Computes the starting global offset (0-based) for a given rank
    implicit none
    integer(idx_k), intent(in) :: global_size, mpi_comm_size, mpi_rank
    integer(idx_k) :: offset
    integer(idx_k) :: base, remainder

    base = global_size / mpi_comm_size
    remainder = mod(global_size, mpi_comm_size)

    if (mpi_rank < remainder) then
      offset = mpi_rank * (base + 1_idx_k)
    else
      offset = remainder * (base + 1_idx_k) + (mpi_rank - remainder) * base
    end if
  end function compute_local_offset

  !--------------------------------------------------------------------------
  ! Distributed Validation Function
  !--------------------------------------------------------------------------
  function distributed_check_result(local_u, g_nn, node_start_idx, nnode_local, tol, rank) result(global_failed)
    ! Checks the local portion of the solution 'local_u' against expected values.
    ! Uses MPI_Allreduce to aggregate results. Returns 0 if passed on all ranks, 1 otherwise.
    implicit none
    real(8), dimension(:), intent(in) :: local_u         ! Local solution array
    integer(idx_k), intent(in) :: g_nn            ! Global grid dimension (NN)
    integer(idx_k), intent(in) :: node_start_idx  ! Starting global index (0-based) for this rank
    integer(idx_k), intent(in) :: nnode_local     ! Number of nodes on this rank
    real(8), intent(in) :: tol             ! Tolerance for comparison
    integer, intent(in) :: rank            ! MPI rank of this process
    integer :: global_failed   ! Result: 0 = pass, 1 = fail

    integer :: local_failed
    integer(idx_k) :: local_idx, global_idx, i, j
    real(8) :: expected_value, diff
    integer :: mpi_ierr

    local_failed = 0 ! Assume success locally initially

    do local_idx = 1_idx_k, nnode_local
      ! Calculate global (0-based) index from local index
      global_idx = node_start_idx + local_idx - 1_idx_k

      ! Calculate global (i, j) coordinates (1-based in range [1, g_nn-1])
      j = global_idx / (g_nn - 1) + 1_idx_k
      i = mod(global_idx, g_nn - 1) + 1_idx_k

      ! Determine expected value based on global (i,j) position
      if (((i == 1_idx_k) .and. (j == 1_idx_k)) .or. &
          ((i == 1_idx_k) .and. (j == (g_nn - 1))) .or. &
          ((i == (g_nn - 1)) .and. (j == 1_idx_k)) .or. &
          ((i == (g_nn - 1)) .and. (j == (g_nn - 1)))) then
        ! Corners
        expected_value = 0.625_8
      else if (((i == 1_idx_k) .and. (j == 2_idx_k)) .or. &
               ((i == 2_idx_k) .and. (j == 1_idx_k)) .or. &
               ((i == 1_idx_k) .and. (j == (g_nn - 2))) .or. &
               ((i == 2_idx_k) .and. (j == (g_nn - 1))) .or. &
               ((i == (g_nn - 2)) .and. (j == 1_idx_k)) .or. &
               ((i == (g_nn - 1)) .and. (j == 2_idx_k)) .or. &
               ((i == (g_nn - 2)) .and. (j == (g_nn - 1))) .or. &
               ((i == (g_nn - 1)) .and. (j == (g_nn - 2)))) then
        ! Horizontally or vertically-adjacent to a corner
        expected_value = 0.4375_8
      else if (((i == 2_idx_k) .and. (j == 2_idx_k)) .or. &
               ((i == 2_idx_k) .and. (j == (g_nn - 2))) .or. &
               ((i == (g_nn - 2)) .and. (j == 2_idx_k)) .or. &
               ((i == (g_nn - 2)) .and. (j == (g_nn - 2)))) then
        ! Diagonally adjacent to a corner
        expected_value = 0.125_8
      else if ((i == 1_idx_k) .or. (j == 1_idx_k) .or. &
               (i == (g_nn - 1)) .or. (j == (g_nn - 1))) then
        ! On some other edge node
        expected_value = 0.3750_8
      else if ((i == 2_idx_k) .or. (j == 2_idx_k) .or. &
               (i == (g_nn - 2)) .or. (j == (g_nn - 2))) then
        ! On some other node that is 1 node from the edge
        expected_value = 0.0625_8
      else
        ! 2 or more nodes from the edge
        expected_value = 0.0_8
      end if

      ! Check if the value matches the expected value within tolerance
      diff = abs(local_u(local_idx) - expected_value)
      if (diff > tol) then
         print *, "Validation Failure on rank ", rank, ": i=", i, ", j=", j, &
                  ", expected=", expected_value, ", actual=", local_u(local_idx), ", diff=", diff
         local_failed = 1
         exit ! Exit loop on first failure for this rank
      end if
    end do

    ! Use MPI_Allreduce to combine validation results (logical OR)
    ! Ranks with local_failed=1 will cause global_failed to become 1.
    call MPI_Allreduce(local_failed, global_failed, 1, MPI_INTEGER, MPI_LOR, MPI_COMM_WORLD, mpi_ierr)
    if (mpi_ierr /= MPI_SUCCESS) then
       print *, "Rank ", rank, ": MPI_Allreduce error in validation!"
       global_failed = 1 ! Force failure indication
    end if

    ! Print final status on root rank
    if (rank == mpi_root) then
       if (global_failed == 0) then
          print *, "Distributed results check PASSED on all ranks!"
       else
          print *, "Distributed results check FAILED on at least one rank!"
       end if
    end if

  end function distributed_check_result

end program jac_distributed
