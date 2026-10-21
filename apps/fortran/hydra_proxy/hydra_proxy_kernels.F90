module hydra_proxy_kernels
!
! Kernels of the strong-scaling proxy: an edge-based finite-volume shape
! (gradients, limiter, upwind flux, boundary fluxes, viscous flux, residual
! smoothing, Runge-Kutta update) with synthetic but bounded arithmetic.  Not
! a solver: the point is the loop structure, the data volumes and the
! dependencies between loops.
!
    implicit none
    private

    public :: copy5, local_dt, zero5, zero15, grad_edge, grad_bnd, grad_scale, &
              lim_edge, lim_finish, flux_edge, flux_bnd, visc_edge, smooth_edge, &
              smooth_node, rk_update, res_norm, sum5

contains

    subroutine copy5(a, b)
        real(8), dimension(5), intent(in) :: a
        real(8), dimension(5), intent(out) :: b

        integer(4) :: k

        do k = 1, 5
            b(k) = a(k)
        end do
    end subroutine

    subroutine local_dt(q, vol, dtl)
        real(8), dimension(5), intent(in) :: q
        real(8), intent(in) :: vol
        real(8), intent(out) :: dtl

        real(8) :: rho, u2, p, c

        rho = max(q(1), 0.1_8)
        u2 = (q(2)**2 + q(3)**2 + q(4)**2) / rho**2
        p = max(0.4_8 * (q(5) - 0.5_8 * rho * u2), 0.01_8)
        c = sqrt(1.4_8 * p / rho)
        dtl = 0.1_8 * vol / (sqrt(u2) + c)
    end subroutine

    subroutine zero5(r)
        real(8), dimension(5), intent(out) :: r

        integer(4) :: k

        do k = 1, 5
            r(k) = 0.0_8
        end do
    end subroutine

    subroutine zero15(g)
        real(8), dimension(15), intent(out) :: g

        integer(4) :: k

        do k = 1, 15
            g(k) = 0.0_8
        end do
    end subroutine

    subroutine grad_edge(q1, q2, ew, g1, g2)
        real(8), dimension(5), intent(in) :: q1, q2
        real(8), dimension(3), intent(in) :: ew
        real(8), dimension(15), intent(inout) :: g1, g2

        integer(4) :: k, c
        real(8) :: d

        do k = 1, 5
            d = 0.5_8 * (q2(k) - q1(k))
            do c = 1, 3
                g1(3 * (k - 1) + c) = g1(3 * (k - 1) + c) + d * ew(c)
                g2(3 * (k - 1) + c) = g2(3 * (k - 1) + c) + d * ew(c)
            end do
        end do
    end subroutine

    subroutine grad_bnd(q, bn, g)
        real(8), dimension(5), intent(in) :: q
        real(8), dimension(3), intent(in) :: bn
        real(8), dimension(15), intent(inout) :: g

        integer(4) :: k, c

        do k = 1, 5
            do c = 1, 3
                g(3 * (k - 1) + c) = g(3 * (k - 1) + c) + 0.1_8 * q(k) * bn(c)
            end do
        end do
    end subroutine

    subroutine grad_scale(vol, g)
        real(8), intent(in) :: vol
        real(8), dimension(15), intent(inout) :: g

        integer(4) :: k

        do k = 1, 15
            g(k) = g(k) / vol
        end do
    end subroutine

    subroutine lim_edge(q1, q2, g1, g2, ew, l1, l2)
        real(8), dimension(5), intent(in) :: q1, q2
        real(8), dimension(15), intent(in) :: g1, g2
        real(8), dimension(3), intent(in) :: ew
        real(8), dimension(5), intent(inout) :: l1, l2

        integer(4) :: k, c
        real(8) :: dq, p1, p2

        ! A smooth limiter: rounding-level gradients must not flip it.
        do k = 1, 5
            dq = abs(q2(k) - q1(k)) + 0.01_8
            p1 = 0.0_8
            p2 = 0.0_8
            do c = 1, 3
                p1 = p1 + g1(3 * (k - 1) + c) * ew(c)
                p2 = p2 + g2(3 * (k - 1) + c) * ew(c)
            end do
            l1(k) = l1(k) + dq / (dq + abs(p1))
            l2(k) = l2(k) + dq / (dq + abs(p2))
        end do
    end subroutine

    subroutine lim_finish(deg, l)
        real(8), intent(in) :: deg
        real(8), dimension(5), intent(inout) :: l

        integer(4) :: k

        do k = 1, 5
            l(k) = l(k) / max(deg, 1.0_8)
        end do
    end subroutine

    subroutine flux_edge(q1, q2, g1, g2, l1, l2, ew, r1, r2)
        real(8), dimension(5), intent(in) :: q1, q2, l1, l2
        real(8), dimension(15), intent(in) :: g1, g2
        real(8), dimension(3), intent(in) :: ew
        real(8), dimension(5), intent(inout) :: r1, r2

        integer(4) :: k, c
        real(8), dimension(5) :: ql, qr, fl, fr
        real(8) :: area, n1, n2, n3, p1, p2
        real(8) :: rhol, ul, vl, wl, pl, hl, unl
        real(8) :: rhor, ur, vr, wr, pr, hr, unr
        real(8) :: sl, sr, ua, va, wa, ha, ca, una, lam

        area = sqrt(ew(1)**2 + ew(2)**2 + ew(3)**2)
        n1 = ew(1) / area
        n2 = ew(2) / area
        n3 = ew(3) / area

        do k = 1, 5
            p1 = 0.0_8
            p2 = 0.0_8
            do c = 1, 3
                p1 = p1 + g1(3 * (k - 1) + c) * ew(c)
                p2 = p2 + g2(3 * (k - 1) + c) * ew(c)
            end do
            ql(k) = q1(k) + 0.05_8 * l1(k) * p1
            qr(k) = q2(k) - 0.05_8 * l2(k) * p2
        end do

        rhol = max(ql(1), 0.1_8)
        ul = ql(2) / rhol
        vl = ql(3) / rhol
        wl = ql(4) / rhol
        pl = max(0.4_8 * (ql(5) - 0.5_8 * rhol * (ul**2 + vl**2 + wl**2)), 0.01_8)
        hl = (ql(5) + pl) / rhol
        unl = ul * n1 + vl * n2 + wl * n3

        rhor = max(qr(1), 0.1_8)
        ur = qr(2) / rhor
        vr = qr(3) / rhor
        wr = qr(4) / rhor
        pr = max(0.4_8 * (qr(5) - 0.5_8 * rhor * (ur**2 + vr**2 + wr**2)), 0.01_8)
        hr = (qr(5) + pr) / rhor
        unr = ur * n1 + vr * n2 + wr * n3

        sl = sqrt(rhol)
        sr = sqrt(rhor)
        ua = (sl * ul + sr * ur) / (sl + sr)
        va = (sl * vl + sr * vr) / (sl + sr)
        wa = (sl * wl + sr * wr) / (sl + sr)
        ha = (sl * hl + sr * hr) / (sl + sr)
        ca = sqrt(max(0.4_8 * (ha - 0.5_8 * (ua**2 + va**2 + wa**2)), 0.01_8))
        una = ua * n1 + va * n2 + wa * n3
        lam = abs(una) + ca

        fl(1) = rhol * unl
        fl(2) = ql(2) * unl + pl * n1
        fl(3) = ql(3) * unl + pl * n2
        fl(4) = ql(4) * unl + pl * n3
        fl(5) = (ql(5) + pl) * unl
        fr(1) = rhor * unr
        fr(2) = qr(2) * unr + pr * n1
        fr(3) = qr(3) * unr + pr * n2
        fr(4) = qr(4) * unr + pr * n3
        fr(5) = (qr(5) + pr) * unr

        do k = 1, 5
            p1 = area * (0.5_8 * (fl(k) + fr(k)) - 0.5_8 * lam * (qr(k) - ql(k)))
            r1(k) = r1(k) + p1
            r2(k) = r2(k) - p1
        end do
    end subroutine

    subroutine flux_bnd(q, bn, r)
        real(8), dimension(5), intent(in) :: q
        real(8), dimension(3), intent(in) :: bn
        real(8), dimension(5), intent(inout) :: r

        real(8) :: a

        a = 0.05_8 * sqrt(bn(1)**2 + bn(2)**2 + bn(3)**2)
        r(1) = r(1) + a * (q(1) - 1.0_8)
        r(2) = r(2) + a * (q(2) - 0.1_8)
        r(3) = r(3) + a * q(3)
        r(4) = r(4) + a * q(4)
        r(5) = r(5) + a * (q(5) - 2.5_8)
    end subroutine

    subroutine visc_edge(q1, q2, ew, r1, r2)
        real(8), dimension(5), intent(in) :: q1, q2
        real(8), dimension(3), intent(in) :: ew
        real(8), dimension(5), intent(inout) :: r1, r2

        integer(4) :: k
        real(8) :: a, d

        a = 0.01_8 * sqrt(ew(1)**2 + ew(2)**2 + ew(3)**2)
        do k = 1, 5
            d = a * (q2(k) - q1(k))
            r1(k) = r1(k) - d
            r2(k) = r2(k) + d
        end do
    end subroutine

    subroutine smooth_edge(r1, r2, s1, s2)
        real(8), dimension(5), intent(in) :: r1, r2
        real(8), dimension(5), intent(inout) :: s1, s2

        integer(4) :: k

        do k = 1, 5
            s1(k) = s1(k) + 0.1_8 * (r2(k) - r1(k))
            s2(k) = s2(k) + 0.1_8 * (r1(k) - r2(k))
        end do
    end subroutine

    subroutine smooth_node(s, r)
        real(8), dimension(5), intent(inout) :: s, r

        integer(4) :: k

        do k = 1, 5
            r(k) = r(k) + s(k)
            s(k) = 0.0_8
        end do
    end subroutine

    subroutine rk_update(qold, res, dtl, vol, alpha, q)
        real(8), dimension(5), intent(in) :: qold, res
        real(8), intent(in) :: dtl, vol, alpha
        real(8), dimension(5), intent(out) :: q

        integer(4) :: k
        real(8) :: a

        a = alpha * dtl / vol
        do k = 1, 5
            q(k) = qold(k) - a * res(k)
        end do
        q(1) = max(q(1), 0.1_8)
        q(5) = max(q(5), 0.1_8)
    end subroutine

    ! The squared residual's sum and maximum, with its node, when on.
    subroutine res_norm(res, idx, on, nrm, rmax, iloc)
        real(8), dimension(5), intent(in) :: res
        integer(4), intent(in) :: idx, on
        real(8), intent(inout) :: nrm, rmax
        integer(4), intent(inout) :: iloc

        real(8) :: r

        if (on == 0) return
        r = res(1)**2 + res(2)**2 + res(3)**2 + res(4)**2 + res(5)**2
        nrm = nrm + r
        if (r > rmax) then
            rmax = r
            iloc = idx
        end if
    end subroutine

    subroutine sum5(q, s)
        real(8), dimension(5), intent(in) :: q
        real(8), dimension(5), intent(inout) :: s

        integer(4) :: k

        do k = 1, 5
            s(k) = s(k) + q(k)
        end do
    end subroutine

end module hydra_proxy_kernels
