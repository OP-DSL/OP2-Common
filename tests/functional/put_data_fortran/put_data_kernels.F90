module put_data_kernels
  implicit none
  private

  public :: copy1, copy3, copy4

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
end module put_data_kernels
