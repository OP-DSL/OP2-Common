! Kernels written the ways f2c translates: loops with negative and run-time
! steps, logical expressions, a function with a result variable, kind
! spellings, upper-case declarations, increments spelt in either case,
! do while, intrinsics, and local arrays.
module corpus_kernels
  implicit none
  private

  public :: steps, logic, kinds, increments, control

contains
  pure function weighted(a, w) result(r)
    real(8), intent(in) :: a, w
    real(8) :: r

    r = a * w + 1.0_8
  end function weighted

  subroutine steps(x, n, y)
    real(8), dimension(4), intent(in) :: x
    integer(4), intent(in) :: n
    real(8), dimension(4), intent(out) :: y
    integer(4) :: i, s

    y = 0.0_8
    do i = 4, 1, -1
      y(i) = y(i) + x(5 - i) * i
    end do
    s = 1 - 2 * mod(n, 2)
    do i = 2 + s, 2 - s, -s
      y(i) = y(i) + weighted(x(i), real(i, 8))
    end do
    do i = 1, 4, 3
      y(i) = y(i) - 0.5_8
    end do
  end subroutine steps

  subroutine logic(x, y)
    real(8), dimension(3), intent(in) :: x
    real(8), dimension(3), intent(out) :: y
    logical :: flag

    flag = x(2) > x(3) .and. .not. x(1) < 0.25_8
    y(1) = 0.0_8
    if (.not. x(1) > 0.5_8) y(1) = 1.0_8
    if (flag) then
      y(2) = 1.0_8
    else if (x(2) == x(3) .or. x(3) /= x(3)) then
      y(2) = 2.0_8
    else
      y(2) = 3.0_8
    end if
    y(3) = x(2)
    if (flag .or. x(3) >= 0.5_8) y(3) = x(1)
  end subroutine logic

  subroutine kinds(x, k, y)
    REAL(KIND=8), DIMENSION(3), INTENT(IN) :: X
    INTEGER(4), INTENT(IN) :: K
    real(8), dimension(3), intent(out) :: y
    double precision :: d
    real(kind=8) :: e
    real(4) :: f
    integer(kind=4) :: m
    integer(8) :: big

    d = X(1) * 2
    e = real(K, 8) / 3.0_8
    f = real(X(2), 4)
    m = int(X(3) * 10.0_8)
    big = int(K, 8) * 3000000000_8
    y(1) = d + e
    y(2) = dble(f) + real(m, 8)
    y(3) = real(mod(big, 7_8), 8)
  end subroutine kinds

  subroutine increments(a, b, w)
    real(8), dimension(3) :: a, b
    real(8), intent(in) :: w

    a(1) = A(1) + w
    A(2) = a(2) - 0.5_8 * w
    a(3) = 2.0_8 * w + a(3)
    B(1) = (b(1) + w) - 1.0_8
    b(2) = b(2) + w * w
    b(3) = B(3) + 1.0_8
  end subroutine increments

  subroutine control(x, k, y)
    real(8), dimension(3), intent(in) :: x
    integer(4), intent(in) :: k
    real(8), dimension(2), intent(out) :: y
    integer(4), parameter :: n = 3
    real(8) :: v(n)
    integer(4) :: i

    v = 0.0_8
    do i = 1, n
      if (x(i) >= 0.1_8) v(i) = sqrt(abs(x(i))) + sign(1.0_8, x(i) - 0.5_8)
    end do
    if (mod(k, 3) == 0) then
      y(1) = min(v(1), v(2), v(3))
    else if (mod(k, 3) == 1) then
      y(1) = max(v(1), v(2)) ** 2
    else
      y(1) = -1.0_8
    end if
    i = 0
    do while (i < k .and. i < 5)
      i = i + 1
    end do
    y(2) = real(i, 8) + v(1) + v(2) + v(3)
  end subroutine control
end module corpus_kernels
