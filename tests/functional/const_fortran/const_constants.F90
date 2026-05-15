module const_constants
    implicit none
    private

    public :: my_const1, my_const4

    real(8), parameter :: my_const1 = 20.1d0
    real(8), dimension(4), parameter :: my_const4 = (/ 30.1d0, 40.2d0, 50.3d0, 60.4d0 /)
end module
