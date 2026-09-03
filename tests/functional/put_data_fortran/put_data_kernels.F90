module put_data_kernels
  implicit none
  private

  public :: copy1, copy3, copy4, gather2, poison

contains
  subroutine copy1(out, in)
    real(8), intent(out) :: out
    real(8), intent(in) :: in

    out = in
  end subroutine copy1

  subroutine copy3(out, in)
    real(8), dimension(3), intent(out) :: out
    real(8), dimension(3), intent(in) :: in
    integer :: d

    do d = 1, 3
      out(d) = in(d)
    end do
  end subroutine copy3

  subroutine copy4(out, in)
    real(8), dimension(4), intent(out) :: out
    real(8), dimension(4), intent(in) :: in
    integer :: d

    do d = 1, 4
      out(d) = in(d)
    end do
  end subroutine copy4

  ! reads both end nodes of an edge, so it touches halo values
  subroutine gather2(out, n0, n1)
    real(8), dimension(2), intent(out) :: out
    real(8), intent(in) :: n0
    real(8), intent(in) :: n1

    out(1) = n0
    out(2) = n1
  end subroutine gather2

  subroutine poison(out)
    real(8), intent(out) :: out

    out = -999.0d0
  end subroutine poison
end module put_data_kernels
