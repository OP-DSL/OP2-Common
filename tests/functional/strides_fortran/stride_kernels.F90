module stride_kernels
    implicit none
    private

    public :: write5, write5_within_kernel

contains
  subroutine write5(dat, val)
    real(8), dimension(5), intent(out) :: dat
    real(8), intent(in) :: val
    real(8) :: power
    integer :: i

    power = 1.0d0
    do i = 1, 5
      dat(i) = val * i * power
      power = power * 10.0d0
    end do
  end subroutine write5

  subroutine write5_within_kernel(dat0, dat1, val)
    real(8), dimension(5), intent(out) :: dat0
    real(8), dimension(5), intent(out) :: dat1
    real(8), intent(in) :: val
    real(8) :: power
    integer :: i

    power = 1.0d0
    do i = 1, 5
      dat0(i) = val * i * power
      power = power * 0.1d0
    end do
    call write5(dat1, val)
  end subroutine write5_within_kernel
end module