program hydra_proxy
!
! A strong-scaling proxy for Hydra-like OP2 codes: many small loops per
! iteration over nodes, edges and six boundary sets, the same kernels called
! on several sets, gradient -> limiter -> flux chains, residual smoothing,
! Runge-Kutta stages and a residual norm.  Halos are exchanged inside every
! stage (q, gradients, limiter, residual per smoothing sweep).  The norm, and
! the residual's maximum with its location, are reduced every monitor
! iterations (and the first, middle and last), through optional arguments
! that are off in between, as Hydra's monitoring is.
!
! The mesh is an nx x ny x nz box of nodes with edges along the axes, each
! rank building its own slab of z planes, so it arrives partitioned.  With
! two chains, a second set of variables goes through the same stages on its
! own dats, as an independent equation set (a turbulence model, say) would.
!
!     hydra_proxy [nx ny nz [iterations [stages [sweeps [chains [monitor]]]]]]
!
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE
    use OP2_FORTRAN_RT_SUPPORT
    use hydra_proxy_kernels

    implicit none

    integer(4) :: nx = 64, ny = 64, nz = 64, iters = 50, stages = 3, sweeps = 2, chains = 1
    integer(4) :: monitor = 1
    integer(4) :: rank, ranks, k0, nk, nnode, nedge, it, s, m, b, i, j, k, l, e, c, ch
    integer(4) :: on, iloc
    integer(4), dimension(6) :: nbnd
    logical :: monitoring
    real(8) :: alpha, nrm, rmax, t0, t1, th
    real(8), dimension(5) :: total
    character(len=32) :: arg

    integer(4), dimension(:), allocatable, target :: e2n
    ! Per chain: q(:, ch) and so on.
    real(8), dimension(:, :), allocatable, target :: q, qold, res, grad, lim, sm
    real(8), dimension(:), allocatable, target :: dtl, vol, deg, ew
    type :: boundary
        integer(4), dimension(:), allocatable :: b2n
        real(8), dimension(:), allocatable :: bn
    end type
    type(boundary), dimension(6) :: bnd

    type(op_set) :: nodes, edges
    type(op_set), dimension(6) :: bsets
    type(op_map) :: p_e2n
    type(op_map), dimension(6) :: p_b2n
    type(op_dat), dimension(2) :: p_q, p_qold, p_res, p_grad, p_lim, p_sm
    type(op_dat) :: p_dtl, p_vol, p_deg, p_ew
    type(op_dat), dimension(6) :: p_bn
    type(op_set) :: dummy_set
    type(op_map) :: dummy_map
    type(op_dat) :: dummy_dat
    character(len=8), dimension(6) :: bname = &
        (/ "bnd_xlo ", "bnd_xhi ", "bnd_ylo ", "bnd_yhi ", "bnd_zlo ", "bnd_zhi " /)

    if (command_argument_count() >= 3) then
        call get_command_argument(1, arg)
        read (arg, *) nx
        call get_command_argument(2, arg)
        read (arg, *) ny
        call get_command_argument(3, arg)
        read (arg, *) nz
    end if
    if (command_argument_count() >= 4) then
        call get_command_argument(4, arg)
        read (arg, *) iters
    end if
    if (command_argument_count() >= 5) then
        call get_command_argument(5, arg)
        read (arg, *) stages
    end if
    if (command_argument_count() >= 6) then
        call get_command_argument(6, arg)
        read (arg, *) sweeps
    end if
    if (command_argument_count() >= 7) then
        call get_command_argument(7, arg)
        read (arg, *) chains
    end if
    chains = max(1, min(chains, 2))
    if (command_argument_count() >= 8) then
        call get_command_argument(8, arg)
        read (arg, *) monitor
    end if
    monitor = max(1, monitor)

    call op_init_base(0, 0)
    call op_profile_start("hydra_proxy")
    call op_mpi_rank(rank)
    ranks = op_mpi_size()

    ! This rank's slab of z planes.
    nk = nz / ranks
    k0 = rank * nk + min(rank, mod(nz, ranks))
    if (rank < mod(nz, ranks)) nk = nk + 1
    nnode = nx * ny * nk

    allocate(q(5 * nnode, 2), qold(5 * nnode, 2), res(5 * nnode, 2), grad(15 * nnode, 2))
    allocate(lim(5 * nnode, 2), sm(5 * nnode, 2))
    allocate(dtl(nnode), vol(nnode), deg(nnode))
    do k = k0, k0 + nk - 1
        do j = 0, ny - 1
            do i = 0, nx - 1
                l = 1 + i + nx * (j + ny * (k - k0))
                do ch = 1, 2
                    q(5 * l - 4, ch) = 1.0_8 + 0.1_8 * sin(0.1_8 * i + ch) * cos(0.13_8 * j)
                    q(5 * l - 3, ch) = 0.1_8 + 0.01_8 * sin(0.07_8 * k)
                    q(5 * l - 2, ch) = 0.01_8 * cos(0.11_8 * i)
                    q(5 * l - 1, ch) = 0.01_8 * sin(0.05_8 * j + ch)
                    q(5 * l, ch) = 2.5_8 + 0.05_8 * cos(0.09_8 * (i + j + k))
                end do
                vol(l) = 1.0_8 + 0.1_8 * sin(0.3_8 * (i + 2 * j + 3 * k))
                deg(l) = 6.0_8
                if (i == 0 .or. i == nx - 1) deg(l) = deg(l) - 1.0_8
                if (j == 0 .or. j == ny - 1) deg(l) = deg(l) - 1.0_8
                if (k == 0 .or. k == nz - 1) deg(l) = deg(l) - 1.0_8
            end do
        end do
    end do
    qold = q
    res = 0.0_8
    grad = 0.0_8
    lim = 0.0_8
    dtl = 0.0_8
    sm = 0.0_8

    ! Edges along the axes from each owned node; the far node of a z edge may
    ! belong to the next rank.  Map entries are global node numbers from 0.
    nedge = (nx - 1) * ny * nk + nx * (ny - 1) * nk + nx * ny * (min(k0 + nk, nz - 1) - k0)
    allocate(e2n(2 * nedge), ew(3 * nedge))
    e = 0
    do k = k0, k0 + nk - 1
        do j = 0, ny - 1
            do i = 0, nx - 1
                l = i + nx * (j + ny * k)
                if (i < nx - 1) call add_edge(l, l + 1, 1)
                if (j < ny - 1) call add_edge(l, l + nx, 2)
                if (k < nz - 1) call add_edge(l, l + nx * ny, 3)
            end do
        end do
    end do

    ! The six faces' owned nodes, with outward normals.
    do b = 1, 6
        nbnd(b) = 0
        allocate(bnd(b)%b2n(nx * ny * nk), bnd(b)%bn(3 * nx * ny * nk))
    end do
    do k = k0, k0 + nk - 1
        do j = 0, ny - 1
            do i = 0, nx - 1
                l = i + nx * (j + ny * k)
                if (i == 0) call add_bnd(1, l, -1.0_8, 0.0_8, 0.0_8)
                if (i == nx - 1) call add_bnd(2, l, 1.0_8, 0.0_8, 0.0_8)
                if (j == 0) call add_bnd(3, l, 0.0_8, -1.0_8, 0.0_8)
                if (j == ny - 1) call add_bnd(4, l, 0.0_8, 1.0_8, 0.0_8)
                if (k == 0) call add_bnd(5, l, 0.0_8, 0.0_8, -1.0_8)
                if (k == nz - 1) call add_bnd(6, l, 0.0_8, 0.0_8, 1.0_8)
            end do
        end do
    end do

    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nedge, edges, "edges")
    call op_decl_map(edges, nodes, 2, e2n, p_e2n, "e2n")
    do b = 1, 6
        call op_decl_set(nbnd(b), bsets(b), trim(bname(b)))
        call op_decl_map(bsets(b), nodes, 1, bnd(b)%b2n, p_b2n(b), trim(bname(b)) // "_n")
        call op_decl_dat(bsets(b), 3, "real(8)", bnd(b)%bn, p_bn(b), trim(bname(b)) // "_normal")
    end do
    do ch = 1, chains
        call op_decl_dat(nodes, 5, "real(8)", q(:, ch), p_q(ch), "q" // achar(48 + ch))
        call op_decl_dat(nodes, 5, "real(8)", qold(:, ch), p_qold(ch), "qold" // achar(48 + ch))
        call op_decl_dat(nodes, 5, "real(8)", res(:, ch), p_res(ch), "res" // achar(48 + ch))
        call op_decl_dat(nodes, 15, "real(8)", grad(:, ch), p_grad(ch), "grad" // achar(48 + ch))
        call op_decl_dat(nodes, 5, "real(8)", lim(:, ch), p_lim(ch), "lim" // achar(48 + ch))
        call op_decl_dat(nodes, 5, "real(8)", sm(:, ch), p_sm(ch), "smooth" // achar(48 + ch))
    end do
    call op_decl_dat(nodes, 1, "real(8)", dtl, p_dtl, "dtl")
    call op_decl_dat(nodes, 1, "real(8)", vol, p_vol, "vol")
    call op_decl_dat(nodes, 1, "real(8)", deg, p_deg, "deg")
    call op_decl_dat(edges, 3, "real(8)", ew, p_ew, "edge_weights")
    call op_partition("", "", dummy_set, dummy_map, dummy_dat)

    if (rank == 0) print '(A,3I6,A,I4,A,I8,A,I3,A,I3,A,I5,A,I2)', " mesh", nx, ny, nz, &
        "  ranks", ranks, "  nodes/rank", nnode, "  stages", stages, "  sweeps", sweeps, &
        "  iterations", iters, "  chains", chains

    call op_timers(t0)
    do it = 1, iters
        do ch = 1, chains
            call op_par_loop_2(copy5, nodes, &
                op_arg_dat(p_q(ch), -1, OP_ID, 5, "real(8)", OP_READ), &
                op_arg_dat(p_qold(ch), -1, OP_ID, 5, "real(8)", OP_WRITE))
        end do
        call op_par_loop_3(local_dt, nodes, &
            op_arg_dat(p_q(1), -1, OP_ID, 5, "real(8)", OP_READ), &
            op_arg_dat(p_vol, -1, OP_ID, 1, "real(8)", OP_READ), &
            op_arg_dat(p_dtl, -1, OP_ID, 1, "real(8)", OP_WRITE))

        do s = 1, stages
            alpha = 1.0_8 / dble(stages - s + 1)

            do ch = 1, chains
                call op_par_loop_1(zero15, nodes, &
                    op_arg_dat(p_grad(ch), -1, OP_ID, 15, "real(8)", OP_WRITE))
                call op_par_loop_5(grad_edge, edges, &
                    op_arg_dat(p_q(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_q(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_ew, -1, OP_ID, 3, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), 1, p_e2n, 15, "real(8)", OP_INC), &
                    op_arg_dat(p_grad(ch), 2, p_e2n, 15, "real(8)", OP_INC))
                do b = 1, 6
                    call op_par_loop_3(grad_bnd, bsets(b), &
                        op_arg_dat(p_q(ch), 1, p_b2n(b), 5, "real(8)", OP_READ), &
                        op_arg_dat(p_bn(b), -1, OP_ID, 3, "real(8)", OP_READ), &
                        op_arg_dat(p_grad(ch), 1, p_b2n(b), 15, "real(8)", OP_INC))
                end do
                call op_par_loop_2(grad_scale, nodes, &
                    op_arg_dat(p_vol, -1, OP_ID, 1, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), -1, OP_ID, 15, "real(8)", OP_RW))

                call op_par_loop_1(zero5, nodes, &
                    op_arg_dat(p_lim(ch), -1, OP_ID, 5, "real(8)", OP_WRITE))
                call op_par_loop_7(lim_edge, edges, &
                    op_arg_dat(p_q(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_q(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), 1, p_e2n, 15, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), 2, p_e2n, 15, "real(8)", OP_READ), &
                    op_arg_dat(p_ew, -1, OP_ID, 3, "real(8)", OP_READ), &
                    op_arg_dat(p_lim(ch), 1, p_e2n, 5, "real(8)", OP_INC), &
                    op_arg_dat(p_lim(ch), 2, p_e2n, 5, "real(8)", OP_INC))
                call op_par_loop_2(lim_finish, nodes, &
                    op_arg_dat(p_deg, -1, OP_ID, 1, "real(8)", OP_READ), &
                    op_arg_dat(p_lim(ch), -1, OP_ID, 5, "real(8)", OP_RW))

                call op_par_loop_1(zero5, nodes, &
                    op_arg_dat(p_res(ch), -1, OP_ID, 5, "real(8)", OP_WRITE))
                call op_par_loop_9(flux_edge, edges, &
                    op_arg_dat(p_q(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_q(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), 1, p_e2n, 15, "real(8)", OP_READ), &
                    op_arg_dat(p_grad(ch), 2, p_e2n, 15, "real(8)", OP_READ), &
                    op_arg_dat(p_lim(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_lim(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_ew, -1, OP_ID, 3, "real(8)", OP_READ), &
                    op_arg_dat(p_res(ch), 1, p_e2n, 5, "real(8)", OP_INC), &
                    op_arg_dat(p_res(ch), 2, p_e2n, 5, "real(8)", OP_INC))
                do b = 1, 6
                    call op_par_loop_3(flux_bnd, bsets(b), &
                        op_arg_dat(p_q(ch), 1, p_b2n(b), 5, "real(8)", OP_READ), &
                        op_arg_dat(p_bn(b), -1, OP_ID, 3, "real(8)", OP_READ), &
                        op_arg_dat(p_res(ch), 1, p_b2n(b), 5, "real(8)", OP_INC))
                end do
                call op_par_loop_5(visc_edge, edges, &
                    op_arg_dat(p_q(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_q(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_ew, -1, OP_ID, 3, "real(8)", OP_READ), &
                    op_arg_dat(p_res(ch), 1, p_e2n, 5, "real(8)", OP_INC), &
                    op_arg_dat(p_res(ch), 2, p_e2n, 5, "real(8)", OP_INC))

                do m = 1, sweeps
                    call op_par_loop_4(smooth_edge, edges, &
                        op_arg_dat(p_res(ch), 1, p_e2n, 5, "real(8)", OP_READ), &
                        op_arg_dat(p_res(ch), 2, p_e2n, 5, "real(8)", OP_READ), &
                        op_arg_dat(p_sm(ch), 1, p_e2n, 5, "real(8)", OP_INC), &
                        op_arg_dat(p_sm(ch), 2, p_e2n, 5, "real(8)", OP_INC))
                    call op_par_loop_2(smooth_node, nodes, &
                        op_arg_dat(p_sm(ch), -1, OP_ID, 5, "real(8)", OP_RW), &
                        op_arg_dat(p_res(ch), -1, OP_ID, 5, "real(8)", OP_RW))
                end do

                call op_par_loop_6(rk_update, nodes, &
                    op_arg_dat(p_qold(ch), -1, OP_ID, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_res(ch), -1, OP_ID, 5, "real(8)", OP_READ), &
                    op_arg_dat(p_dtl, -1, OP_ID, 1, "real(8)", OP_READ), &
                    op_arg_dat(p_vol, -1, OP_ID, 1, "real(8)", OP_READ), &
                    op_arg_gbl(alpha, 1, "real(8)", OP_READ), &
                    op_arg_dat(p_q(ch), -1, OP_ID, 5, "real(8)", OP_WRITE))
            end do
        end do

        monitoring = it == 1 .or. mod(it, monitor) == 0 .or. it == iters / 2 .or. it == iters
        on = merge(1, 0, monitoring)
        nrm = 0.0_8
        rmax = 0.0_8
        iloc = -1
        call op_par_loop_6(res_norm, nodes, &
            op_arg_dat(p_res(1), -1, OP_ID, 5, "real(8)", OP_READ), &
            op_arg_idx(-1, OP_ID), &
            op_arg_gbl(on, 1, "integer(4)", OP_READ), &
            op_opt_arg_gbl(monitoring, nrm, 1, "real(8)", OP_INC), &
            op_opt_arg_gbl(monitoring, rmax, 1, "real(8)", OP_MAX), &
            op_arg_info(iloc, 1, "integer(4)", 5))
        if (rank == 0 .and. monitoring .and. (it == 1 .or. mod(it, 10) == 0)) &
            print '(A,I6,A,ES22.14,A,ES22.14,A,I9)', " iteration", it, "  residual", sqrt(nrm), &
                "  max", sqrt(rmax), " at", iloc
        ! The second half's time, clear of compiles and tuning: the norm has
        ! synchronised.
        if (it == iters / 2) call op_timers(th)
    end do
    call op_timers(t1)

    do ch = 1, chains
        total = 0.0_8
        call op_par_loop_2(sum5, nodes, &
            op_arg_dat(p_q(ch), -1, OP_ID, 5, "real(8)", OP_READ), &
            op_arg_gbl(total, 5, "real(8)", OP_INC))
        if (rank == 0) print '(A,5ES22.14)', " checksum", total
    end do
    if (rank == 0) then
        print '(A,F10.3,A,F10.2,A)', " time", t1 - t0, " s, ", 1.0e6_8 * (t1 - t0) / iters, &
            " us/iteration"
        print '(A,F10.2,A)', " steady", 1.0e6_8 * (t1 - th) / (iters - iters / 2), &
            " us/iteration (second half)"
    end if
    call op_timing_output()
    call op_profile_end()
    call op_profile_output()
    call op_exit()

contains

    subroutine add_edge(n1, n2, axis)
        integer(4), intent(in) :: n1, n2, axis

        e = e + 1
        e2n(2 * e - 1) = n1
        e2n(2 * e) = n2
        do c = 1, 3
            ew(3 * (e - 1) + c) = 0.0_8
        end do
        ew(3 * (e - 1) + axis) = 1.0_8 + 0.1_8 * sin(0.37_8 * dble(n1))
    end subroutine

    subroutine add_bnd(face, node, x, y, z)
        integer(4), intent(in) :: face, node
        real(8), intent(in) :: x, y, z

        nbnd(face) = nbnd(face) + 1
        bnd(face)%b2n(nbnd(face)) = node
        bnd(face)%bn(3 * nbnd(face) - 2) = x
        bnd(face)%bn(3 * nbnd(face) - 1) = y
        bnd(face)%bn(3 * nbnd(face)) = z
    end subroutine

end program hydra_proxy
