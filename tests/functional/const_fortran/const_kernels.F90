module const_kernels
    use const_constants

    implicit none
    private

    public :: consts1, consts4, reverse4, not_greater4, twice4, kinds4

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

  ! Kinds by number, by standard name and by the application's own name.
  subroutine kinds4(dat)
    use, intrinsic :: iso_c_binding, only: c_int
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use const_kinds, only: rk
    real(rk), dimension(4), intent(inout) :: dat
    double precision :: half
    real(kind=real64) :: whole
    integer(int64) :: big
    integer(kind=c_int) :: three

    half = 0.5_rk
    whole = real(int(dat(1), 8), rk)
    big = int(dat(1), int64) + 5_8
    three = 3
    dat(1) = half * dat(1)
    dat(2) = whole
    dat(3) = real(big, 8)
    dat(4) = real(three) * 2.0_RK
  end subroutine kinds4
end module