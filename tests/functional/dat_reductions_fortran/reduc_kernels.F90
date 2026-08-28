module reduc_kernels
    implicit none
    private

    public :: indirect_dat1_inc, indirect_dat3_inc, direct_dat1_inc, direct_dat4_inc
    public :: indirect_inc_reduce

contains
  subroutine indirect_dat1_inc(n0i, n1i, er)
    real(4), intent(inout) :: n0i
    real(4), intent(inout) :: n1i
    real(4), intent(in)    :: er

    n0i = n0i + er
    n1i = n1i + er
  end subroutine indirect_dat1_inc

  ! Indirect increment combined with a global reduction.  The reduction is
  ! what makes the atomics schedule split owned work out from the exec halo,
  ! so this is the loop that exercises all three execution sections.
  subroutine indirect_inc_reduce(n0i, n1i, er, total)
    real(4), intent(inout) :: n0i
    real(4), intent(inout) :: n1i
    real(4), intent(in)    :: er
    real(8), intent(inout) :: total

    n0i = n0i + er
    n1i = n1i + er
    total = total + 2.0_8 * dble(er)
  end subroutine indirect_inc_reduce

  subroutine indirect_dat3_inc(n0i, n1i, er)
    real(4), dimension(3), intent(inout) :: n0i
    real(4), dimension(3), intent(inout) :: n1i
    real(4), dimension(4), intent(in)    :: er
    integer :: d

    do d = 1, 3
      n0i(d) = n0i(d) + er(d)
      n1i(d) = n1i(d) + er(d)
    end do
  end subroutine indirect_dat3_inc

  subroutine direct_dat1_inc(dr, di)
    real(4), intent(in)    :: dr
    real(4), intent(inout) :: di

    di = di + (dr + 3.25)
  end subroutine direct_dat1_inc

  subroutine direct_dat4_inc(dr, di)
    real(4), dimension(4), intent(in) :: dr
    real(4), dimension(4), intent(inout) :: di
    integer :: d

    do d = 1, 4
      di(d) = di(d) + (dr(d) + 1.325 * (d - 1))
    end do
  end subroutine direct_dat4_inc
end module