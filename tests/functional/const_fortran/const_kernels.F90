module const_kernels
    use const_constants

    implicit none
    private

    public :: consts1, consts4, reverse4, not_greater4, twice4

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

  ! Reverse the components with a literal negative step, then negate every
  ! other one from the last with a negative step known only at run time.
  subroutine reverse4(dat)
    real(8), dimension(4), intent(inout) :: dat
    real(8) :: swap
    integer :: d, step

    do d = 4, 3, -1
      swap = dat(d)
      dat(d) = dat(5 - d)
      dat(5 - d) = swap
    end do

    step = -2
    do d = 4, 1, step
      dat(d) = -dat(d)
    end do
  end subroutine reverse4

  ! .NOT. binds looser than a comparison: zero every component not above 45.
  subroutine not_greater4(dat)
    real(8), dimension(4), intent(inout) :: dat
    integer :: d

    do d = 1, 4
      if (.not. dat(d) > 45.0d0) dat(d) = 0.0d0
    end do
  end subroutine not_greater4

  ! A function that returns through a result variable.
  real(8) function twice(v) result(r)
    real(8), intent(in) :: v

    r = 2.0d0 * v
  end function twice

  subroutine twice4(dat)
    real(8), dimension(4), intent(inout) :: dat
    integer :: d

    do d = 1, 4
      dat(d) = twice(dat(d))
    end do
  end subroutine twice4
end module