module smemproxy_kernels
    implicit none
    private

    public :: edge_scatter

contains
  ! Per-edge gather/combine/scatter, ported from the atomics proxy's
  ! edge_scatter_compute.  Both endpoints of the edge receive the same three
  ! weighted differences per component, which is what makes this a
  ! high-intensity indirect increment: 6 * ncomp increments per edge.
  !
  ! out is shaped (component, direction) with direction fastest, matching the
  ! proxy's (c * 3 + d) layout.
  subroutine edge_scatter(in1, in2, coef, out1, out2)
    real(8), dimension(7),  intent(in)    :: in1
    real(8), dimension(7),  intent(in)    :: in2
    real(8), dimension(3),  intent(in)    :: coef
    real(8), dimension(21), intent(inout) :: out1
    real(8), dimension(21), intent(inout) :: out2

    integer(4) :: c, base
    real(8) :: du, du0, du1, du2

    do c = 1, 7
      du = 0.5_8 * (in2(c) - in1(c))
      du0 = coef(1) * du
      du1 = coef(2) * du
      du2 = coef(3) * du

      base = (c - 1) * 3

      out1(base + 1) = out1(base + 1) + du0
      out1(base + 2) = out1(base + 2) + du1
      out1(base + 3) = out1(base + 3) + du2

      out2(base + 1) = out2(base + 1) + du0
      out2(base + 2) = out2(base + 2) + du1
      out2(base + 3) = out2(base + 3) + du2
    end do
  end subroutine edge_scatter
end module smemproxy_kernels
