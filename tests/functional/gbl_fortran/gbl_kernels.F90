module gbl_kernels
    implicit none
    private

    public :: read1, read5, inc1, inc5, min1, min5, max1, max5

contains
  subroutine read1(dat, g)
    real(8), intent(out) :: dat
    real(8), intent(in)  :: g

    dat = g
  end subroutine read1

  subroutine read5(dat, g)
    real(8), dimension(5), intent(out) :: dat
    real(8), dimension(5), intent(in)  :: g
    integer :: i

    do i = 1, 5
      dat(i) = g(i)
    end do
  end subroutine read5

  subroutine inc1(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = g + dat
  end subroutine inc1

  subroutine inc5(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = g(i) + dat(i)
    end do
  end subroutine inc5

  subroutine min1(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = min(g, dat)
  end subroutine min1

  subroutine min5(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = min(g(i), dat(i))
    end do
  end subroutine min5

  subroutine max1(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = max(g, dat)
  end subroutine max1

  subroutine max5(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = max(g(i), dat(i))
    end do
  end subroutine max5
end module