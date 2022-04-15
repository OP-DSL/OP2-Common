module airfoil_kernels
    use airfoil_constants

    implicit none
    private

    public :: save_soln, adt_calc, res_calc, bres_calc, update

contains

    subroutine save_soln(q, qold)
        real(8), dimension(4), intent(in) :: q
        real(8), dimension(4), intent(out) :: qold

        integer(4) :: i

        do i = 1, 4
            qold(i) = q(i)
        end do
    end subroutine

    subroutine adt_calc(x1, x2, x3, x4, q, adt)
        real(8), dimension(2), intent(in) :: x1, x2, x3, x4
        real(8), dimension(4), intent(in) :: q
        real(8), intent(out) :: adt

        real(8) :: dx, dy, ri, u, v, c

        ri = 1.0 / q(1)

        u = ri * q(2)
        v = ri * q(3)

        c = sqrt(gam * gm1 * (ri * q(4) - 0.5 * (u * u + v * v)))

        dx = x2(1) - x1(1)
        dy = x2(2) - x1(2)

        adt = abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)

        dx = x3(1) - x2(1)
        dy = x3(2) - x2(2)

        adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)

        dx = x4(1) - x3(1)
        dy = x4(2) - x3(2)

        adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)

        dx = x1(1) - x4(1)
        dy = x1(2) - x4(2)

        adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
        adt = adt / cfl
    end subroutine

    subroutine res_calc(x1, x2, q1, q2, adt1, adt2, res1, res2)
        real(8), dimension(2), intent(in) :: x1, x2
        real(8), dimension(4), intent(in) :: q1, q2
        real(8), intent(in) :: adt1, adt2
        real(8), dimension(4), intent(inout) :: res1, res2

        real(8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f

        dx = x1(1) - x2(1)
        dy = x1(2) - x2(2)

        ri = 1.0 / q1(1)
        p1 = gm1 * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))

        vol1 = ri * (q1(2) * dy - q1(3) * dx)

        ri = 1.0 / q2(1)
        p2 = gm1 * (q2(4) - 0.5 * ri * (q2(2) * q2(2) + q2(3) * q2(3)))

        vol2 = ri * (q2(2) * dy - q2(3) * dx)

        mu = 0.5 * (adt1 + adt2) * eps
        f = 0.5 * (vol1 * q1(1) + vol2 * q2(1)) + mu * (q1(1) - q2(1))

        res1(1) = res1(1) + f
        res2(1) = res2(1) - f

        f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * q2(2) + p2 * dy) + mu * (q1(2) - q2(2))

        res1(2) = res1(2) + f
        res2(2) = res2(2) - f

        f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * q2(3) - p2 * dx) + mu * (q1(3) - q2(3))

        res1(3) = res1(3) + f
        res2(3) = res2(3) - f

        f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (q2(4) + p2)) + mu * (q1(4) - q2(4))

        res1(4) = res1(4) + f
        res2(4) = res2(4) - f
    end subroutine

    subroutine bres_calc(x1, x2, q1, adt1, res1, bound)
        real(8), dimension(2), intent(in) :: x1, x2
        real(8), dimension(4), intent(in) :: q1
        real(8), intent(in) :: adt1
        real(8), dimension(4), intent(inout) :: res1
        integer(4), intent(in) :: bound

        real(8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f

        dx = x1(1) - x2(1)
        dy = x1(2) - x2(2)

        ri = 1.0 / q1(1)
        p1 = gm1 * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))

        if (bound == 1) then
            res1(2) = res1(2) + p1 * dy
            res1(3) = res1(3) - p1 * dx

            return
        end if

        vol1 = ri * (q1(2) * dy - q1(3) * dx)
        ri = 1.0 / qinf(1)
        p2 = gm1 * (qinf(4) - 0.5 * ri * (qinf(2) * qinf(2) + qinf(3) * qinf(3)))

        vol2 = ri * (qinf(2) * dy - qinf(3) * dx)
        mu = adt1 * eps

        f = 0.5 * (vol1 * q1(1) + vol2 * qinf(1)) + mu * (q1(1) - qinf(1))
        res1(1) = res1(1) + f

        f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * qinf(2) + p2 * dy) + mu * (q1(2) - qinf(2))
        res1(2) = res1(2) + f

        f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * qinf(3) - p2 * dx) + mu * (q1(3) - qinf(3))
        res1(3) = res1(3) + f

        f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (qinf(4) + p2)) + mu * (q1(4) - qinf(4))
        res1(4) = res1(4) + f
    end subroutine

    subroutine update(qold, q, res, adt, rms)
        real(8), dimension(4), intent(in) :: qold
        real(8), dimension(4), intent(out) :: q
        real(8), dimension(4), intent(inout) :: res
        real(8), intent(in) :: adt
        real(8), dimension(2), intent(inout) :: rms

        real(8) :: del, adti
        integer(4) :: i

        adti = 1.0 / adt

        do i = 1, 4
            del = adti * res(i)
            q(i) = qold(i) - del
            res(i) = 0.0

            rms(2) = rms(2) + del * del
        end do
    end subroutine

end module
