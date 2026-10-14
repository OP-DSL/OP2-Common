module rw_kernels
    implicit none
    private

    public :: rw_count, rw_mixed, rw_dims, inc_dims

contains
  ! Count each edge at both of its nodes and at its hub, through read-write
  ! arguments on one dat, so a lost update shows up exactly.
  subroutine rw_count(c0, c1, ch)
    integer(4), intent(inout) :: c0
    integer(4), intent(inout) :: c1
    integer(4), intent(inout) :: ch

    c0 = c0 + 1
    c1 = c1 + 1
    ch = ch + 1
  end subroutine rw_count

  ! Read-write, increment and read arguments on updated dats, an optional
  ! increment and a global reduction.  The far value is only compared, so the
  ! result does not depend on the order conflicting edges run in.  The
  ! optional increment is touched only when use_d says it is active.
  subroutine rw_mixed(a0, a1, a_far, b0, b1, dh, er, use_d, total)
    real(8), dimension(2), intent(inout) :: a0
    real(8), dimension(2), intent(inout) :: a1
    real(8), dimension(2), intent(in)    :: a_far
    real(8), intent(inout) :: b0
    real(8), intent(inout) :: b1
    real(8), intent(inout) :: dh
    real(8), intent(in)    :: er
    integer(4), intent(in) :: use_d
    real(8), intent(inout) :: total

    if (a_far(1) >= 0.0_8) then
      a0(1) = a0(1) + er
      a1(1) = a1(1) + er
    end if
    a0(2) = a0(2) + 2.0_8 * er
    a1(2) = a1(2) + 2.0_8 * er

    b0 = b0 + er
    b1 = b1 + er
    if (use_d == 1) dh = dh + 1.0_8
    total = total + er
  end subroutine rw_mixed

  ! Read-write and increment updates on dats whose dimensions are given only
  ! at run time, from a direct read of another such dat.
  subroutine rw_dims(w0, w1, v0, v1, ew)
    real(8), dimension(2), intent(inout) :: w0
    real(8), dimension(2), intent(inout) :: w1
    real(8), dimension(2), intent(inout) :: v0
    real(8), dimension(2), intent(inout) :: v1
    real(8), dimension(2), intent(in)    :: ew

    w0(1) = w0(1) + ew(1)
    w0(2) = w0(2) + ew(2)
    w1(1) = w1(1) + ew(1)
    w1(2) = w1(2) + ew(2)
    v0(1) = v0(1) + ew(1)
    v0(2) = v0(2) + ew(2)
    v1(1) = v1(1) + ew(1)
    v1(2) = v1(2) + ew(2)
  end subroutine rw_dims

  ! The increments alone, so chunk colouring can compute every element
  ! of a chunk at once.
  subroutine inc_dims(u0, u1, ew)
    real(8), dimension(2), intent(inout) :: u0
    real(8), dimension(2), intent(inout) :: u1
    real(8), dimension(2), intent(in)    :: ew

    u0(1) = u0(1) + ew(1)
    u0(2) = u0(2) + ew(2)
    u1(1) = u1(1) + ew(1)
    u1(2) = u1(2) + ew(2)
  end subroutine inc_dims
end module
