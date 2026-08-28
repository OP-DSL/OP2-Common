program smemproxy
!
! OP2 port of the atomics proxy's edge-scatter kernel, reading the same edge
! maps.  The proxy exists because this access shape - many edges scattering
! into shared nodes - is where global atomic throughput becomes the limit, so
! this is the high-intensity counterpart to Airfoil's res_calc for evaluating
! hierarchical shared-memory atomics.
!
! Mesh path comes from PROXY_MESH so it does not collide with the OP_* options
! op_set_args parses out of the command line.
!
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    use smemproxy_kernels

    implicit none

    integer(4), parameter :: ncomp = 7
    integer(4), parameter :: nout = ncomp * 3
    real(8), parameter :: tol = 1.0e-9_8

    character(len=1024) :: mesh_path
    integer(4) :: path_len, path_status
    integer(4) :: nnode, nedge
    integer(4) :: e, c, n1, n2, base, failures
    integer(4) :: iter, niter
    real(8) :: du, du0, du1, du2, diff

    integer(4), dimension(:), allocatable, target :: e2n
    real(8), dimension(:), allocatable, target :: node_in, edge_coef, node_out
    real(8), dimension(:), allocatable :: fetched, expected

    type(op_set) :: nodes, edges
    type(op_map) :: p_e2n
    type(op_dat) :: p_in, p_coef, p_out

    call op_init(0)

    call get_environment_variable("PROXY_MESH", mesh_path, path_len, path_status)
    if (path_status /= 0 .or. path_len == 0) then
      mesh_path = "edges.txt"
    end if

    call read_edges(trim(mesh_path), nnode, nedge, e2n)
    write(*,'(A,A)')  " mesh:  ", trim(mesh_path)
    write(*,'(A,I0,A,I0)') " nodes: ", nnode, "   edges: ", nedge

    niter = 1
    call get_iterations(niter)

    allocate(node_in(ncomp * nnode))
    allocate(edge_coef(3 * nedge))
    allocate(node_out(nout * nnode))

    ! Deterministic data, so the reference below is reproducible.
    do e = 1, nnode
      do c = 1, ncomp
        node_in((e - 1) * ncomp + c) = &
          dble(mod(e * 7 + c * 13, 101)) * 0.01_8 + 0.5_8
      end do
    end do

    do e = 1, nedge
      edge_coef((e - 1) * 3 + 1) = dble(mod(e * 3, 17)) * 0.1_8 + 0.25_8
      edge_coef((e - 1) * 3 + 2) = dble(mod(e * 5, 19)) * 0.1_8 + 0.35_8
      edge_coef((e - 1) * 3 + 3) = dble(mod(e * 11, 23)) * 0.1_8 + 0.45_8
    end do

    node_out = 0.0_8

    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nedge, edges, "edges")

    call op_decl_map(edges, nodes, 2, e2n, p_e2n, "edge_to_node")

    call op_decl_dat(nodes, ncomp, "real(8)", node_in,   p_in,   "node_in")
    call op_decl_dat(edges, 3,     "real(8)", edge_coef, p_coef, "edge_coef")
    call op_decl_dat(nodes, nout,  "real(8)", node_out,  p_out,  "node_out")

    call op_profile_start("smemproxy")
    call op_profile_enter("Main computation")

    do iter = 1, niter
      call op_par_loop_5(edge_scatter, edges, &
        op_arg_dat(p_in,   1, p_e2n, ncomp, "real(8)", OP_READ), &
        op_arg_dat(p_in,   2, p_e2n, ncomp, "real(8)", OP_READ), &
        op_arg_dat(p_coef, -1, OP_ID, 3,    "real(8)", OP_READ), &
        op_arg_dat(p_out,  1, p_e2n, nout,  "real(8)", OP_INC),  &
        op_arg_dat(p_out,  2, p_e2n, nout,  "real(8)", OP_INC))
    end do

    call op_profile_end()
    call op_profile_output()

    ! Host reference for one sweep, scaled by the iteration count.
    allocate(expected(nout * nnode))
    expected = 0.0_8

    do e = 1, nedge
      n1 = e2n((e - 1) * 2 + 1)
      n2 = e2n((e - 1) * 2 + 2)

      do c = 1, ncomp
        du = 0.5_8 * (node_in((n2 - 1) * ncomp + c) - &
                      node_in((n1 - 1) * ncomp + c))
        du0 = edge_coef((e - 1) * 3 + 1) * du
        du1 = edge_coef((e - 1) * 3 + 2) * du
        du2 = edge_coef((e - 1) * 3 + 3) * du

        base = (c - 1) * 3
        expected((n1 - 1) * nout + base + 1) = &
          expected((n1 - 1) * nout + base + 1) + du0
        expected((n1 - 1) * nout + base + 2) = &
          expected((n1 - 1) * nout + base + 2) + du1
        expected((n1 - 1) * nout + base + 3) = &
          expected((n1 - 1) * nout + base + 3) + du2

        expected((n2 - 1) * nout + base + 1) = &
          expected((n2 - 1) * nout + base + 1) + du0
        expected((n2 - 1) * nout + base + 2) = &
          expected((n2 - 1) * nout + base + 2) + du1
        expected((n2 - 1) * nout + base + 3) = &
          expected((n2 - 1) * nout + base + 3) + du2
      end do
    end do

    expected = expected * dble(niter)

    allocate(fetched(nout * nnode))
    call op_fetch_data(p_out, fetched)

    failures = 0
    do e = 1, nout * nnode
      diff = abs(fetched(e) - expected(e))
      if (diff > tol * max(1.0_8, abs(expected(e)))) then
        failures = failures + 1
        if (failures <= 5) then
          write(*,*) "ERROR at ", e, " got ", fetched(e), " want ", expected(e)
        end if
      end if
    end do

    if (failures == 0) then
      write(*,*) "Test PASSED"
    else
      write(*,*) "Test FAILED:", failures, "mismatches"
    end if

    call op_exit()

    if (failures /= 0) error stop 1

contains

    subroutine get_iterations(iters)
      integer(4), intent(out) :: iters
      character(len=64) :: value
      integer(4) :: vlen, vstatus, ios

      iters = 1
      call get_environment_variable("PROXY_ITERS", value, vlen, vstatus)
      if (vstatus == 0 .and. vlen > 0) then
        read(value, *, iostat=ios) iters
        if (ios /= 0 .or. iters < 1) iters = 1
      end if
    end subroutine get_iterations

    ! Edge map: one "n1 n2" pair of 0-based node indices per line.  OP2's
    ! Fortran API defaults to 1-based maps, so shift on the way in.
    subroutine read_edges(path, out_nnode, out_nedge, map)
      character(len=*), intent(in) :: path
      integer(4), intent(out) :: out_nnode
      integer(4), intent(out) :: out_nedge
      integer(4), dimension(:), allocatable, intent(out), target :: map

      integer(4), parameter :: unit_id = 41
      integer(8) :: fsize, pos
      integer(4) :: ios, value, count, maxnode
      logical :: in_number
      character(len=:), allocatable :: buf
      character :: ch

      ! One stream read and a hand-rolled integer scan.  List-directed reads
      ! of a 24M-line map take minutes; this takes seconds.
      open(unit_id, file=path, access="stream", form="unformatted", &
           status="old", action="read", iostat=ios)
      if (ios /= 0) then
        write(*,*) "ERROR: cannot open mesh file ", path
        error stop 1
      end if

      inquire(unit_id, size=fsize)
      allocate(character(len=fsize) :: buf)
      read(unit_id) buf
      close(unit_id)

      ! Every line holds two indices, so this bounds the pair count.
      allocate(map(2 * (fsize / 4 + 2)))

      count = 0
      value = 0
      maxnode = -1
      in_number = .false.

      do pos = 1, fsize
        ch = buf(pos:pos)
        if (ch >= '0' .and. ch <= '9') then
          value = value * 10 + (iachar(ch) - iachar('0'))
          in_number = .true.
        else if (in_number) then
          count = count + 1
          map(count) = value + 1
          if (value > maxnode) maxnode = value
          value = 0
          in_number = .false.
        end if
      end do

      if (in_number) then
        count = count + 1
        map(count) = value + 1
        if (value > maxnode) maxnode = value
      end if

      deallocate(buf)

      if (mod(count, 2) /= 0) then
        write(*,*) "ERROR: mesh file has an odd index count"
        error stop 1
      end if

      out_nedge = count / 2
      out_nnode = maxnode + 1
    end subroutine read_edges

end program smemproxy
