module airfoil_constants
    implicit none
    private

    public :: gam, gm1, cfl, eps, mach, alpha, qinf

    real(8), parameter :: gam = 1.4_8
    real(8), parameter :: gm1 = gam - 1.0_8
    real(8), parameter :: cfl = 0.9_8
    real(8), parameter :: eps = 0.05_8

    real(8), parameter :: mach = 0.4_8
    real(8), parameter :: alpha = 3.0_8 * atan(1.0_8) / 45.0_8

    real(8), parameter :: p = 1.0_8
    real(8), parameter :: r = 1.0_8
    real(8), parameter :: u = sqrt(gam * p / r) * mach
    real(8), parameter :: e = p / (r * gm1) + 0.5_8 * u**2

    real(8), dimension(4), parameter :: qinf = (/r, r * u, 0.0_8, r * e/)
end module
