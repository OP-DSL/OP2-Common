attributes (device) SUBROUTINE adt_calc(x1,x2,x3,x4,q,adt)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(*) :: x1
  REAL(kind=8), DIMENSION(*) :: x2
  REAL(kind=8), DIMENSION(*) :: x3
  REAL(kind=8), DIMENSION(*) :: x4
  REAL(kind=8), DIMENSION(*) :: q
  REAL(kind=8) :: adt
  REAL(kind=8) :: dx,dy,ri,u,v,c

  ri = 1.0 / q(1)
  u = ri * q(2)
  v = ri * q(3)
  c = sqrt(gam * gm1 * (ri * q(4) - 0.5 * (u * u + v * v)))
  dx = x2(1) - x1(1)
  dy = x2(2) - x1(2)
  adt = abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
  dx = x3(1) - x2(1)
  dy = x3(2) - x2(2)
  adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
  dx = x4(1) - x3(1)
  dy = x4(2) - x3(2)
  adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
  dx = x1(1) - x4(1)
  dy = x1(2) - x4(2)
  adt = adt + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
  adt = adt / cfl
END SUBROUTINE
