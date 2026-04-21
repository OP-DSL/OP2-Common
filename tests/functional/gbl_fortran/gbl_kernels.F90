module gbl_kernels
    implicit none
    private

    public :: read1_k, read5_k, inc1_k, inc5_k, min1_k, min5_k, max1_k, max5_k

contains
  subroutine read1_k(dat, g)
    real(8), intent(out) :: dat
    real(8), intent(in)  :: g

    dat = g
  end subroutine read1_k

  subroutine read5_k(dat, g)
    real(8), dimension(5), intent(out) :: dat
    real(8), dimension(5), intent(in)  :: g
    integer :: i

    do i = 1, 5
      dat(i) = g(i)
    end do
  end subroutine read5_k

  subroutine inc1_k(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = g + dat
  end subroutine inc1_k

  subroutine inc5_k(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = g(i) + dat(i)
    end do
  end subroutine inc5_k

  subroutine min1_k(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = min(g, dat)
  end subroutine min1_k

  subroutine min5_k(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = min(g(i), dat(i))
    end do
  end subroutine min5_k

  subroutine max1_k(dat, g)
    real(8), intent(in)    :: dat
    real(8), intent(inout) :: g

    g = max(g, dat)
  end subroutine max1_k

  subroutine max5_k(dat, g)
    real(8), dimension(5), intent(in)    :: dat
    real(8), dimension(5), intent(inout) :: g
    integer :: i

    do i = 1, 5
      g(i) = max(g(i), dat(i))
    end do
  end subroutine max5_k
end module