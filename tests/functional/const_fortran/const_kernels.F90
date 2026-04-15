module const_kernels
    use const_constants

    implicit none
    private

    public :: consts1, consts4

contains
  subroutine consts1(dat)
    real(8), intent(out) :: dat

    dat = my_const1
  end subroutine consts1

  subroutine consts4(dat)
    real(8), dimension(4), intent(out) :: dat
    integer :: d

    do d = 1, 4
      dat(d) = my_const4(d)
    end do
  end subroutine consts4
end module