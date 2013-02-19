attributes (device) SUBROUTINE res_calc(x1,x2,q1,q2,adt1,adt2,res1,res2)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(2) :: x1
  REAL(kind=8), DIMENSION(2) :: x2
  REAL(kind=8), DIMENSION(4) :: q1
  REAL(kind=8), DIMENSION(4) :: q2
  REAL(kind=8) :: adt1
  REAL(kind=8) :: adt2
  REAL(kind=8), DIMENSION(4) :: res1
  REAL(kind=8), DIMENSION(4) :: res2
  REAL(kind=8) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f

  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0 / q1(1)
  p1 = gm1 * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))
  vol1 = ri * (q1(2) * dy - q1(3) * dx)
  ri = 1.0 / q2(1)
  p2 = gm1 * (q2(4) - 0.5 * ri * (q2(2) * q2(2) + q2(3) * q2(3)))
  vol2 = ri * (q2(2) * dy - q2(3) * dx)
  mu = 0.5 * (adt1 + adt2) * eps
  f = 0.5 * (vol1 * q1(1) + vol2 * q2(1)) + mu * (q1(1) - q2(1))
  res1(1) = res1(1) + f
  res2(1) = res2(1) - f
  f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * q2(2) + p2 * dy) + mu * (q1(2) - q2(2))
  res1(2) = res1(2) + f
  res2(2) = res2(2) - f
  f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * q2(3) - p2 * dx) + mu * (q1(3) - q2(3))
  res1(3) = res1(3) + f
  res2(3) = res2(3) - f
  f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (q2(4) + p2)) + mu * (q1(4) - q2(4))
  res1(4) = res1(4) + f
  res2(4) = res2(4) - f
END SUBROUTINE
