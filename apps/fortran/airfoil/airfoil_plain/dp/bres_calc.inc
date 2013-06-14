SUBROUTINE bres_calc(x1,x2,q1,adt1,res1,bound)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(2) :: x1
  REAL(kind=8), DIMENSION(2) :: x2
  REAL(kind=8), DIMENSION(4) :: q1
  REAL(kind=8) :: adt1
  REAL(kind=8), DIMENSION(4) :: res1
  INTEGER(kind=4) :: bound
  REAL(kind=8) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f

  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0 / q1(1)
  p1 = gm1 * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))

  IF (bound .EQ. 1) THEN
    res1(2) = res1(2) + p1 * dy
    res1(3) = res1(3) + -(p1 * dx)
  ELSE
    vol1 = ri * (q1(2) * dy - q1(3) * dx)
    ri = 1.0 / qinf(1)
    p2 = gm1 * (qinf(4) - 0.5 * ri * (qinf(2) * qinf(2) + qinf(3) * qinf(3)))
    vol2 = ri * (qinf(2) * dy - qinf(3) * dx)
    mu = adt1 * eps
    f = 0.5 * (vol1 * q1(1) + vol2 * qinf(1)) + mu * (q1(1) - qinf(1))
    res1(1) = res1(1) + f
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * qinf(2) + p2 * dy) + mu * (q1(2) - qinf(2))
    res1(2) = res1(2) + f
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * qinf(3) - p2 * dx) + mu * (q1(3) - qinf(3))
    res1(3) = res1(3) + f
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (qinf(4) + p2)) + mu * (q1(4) - qinf(4))
    res1(4) = res1(4) + f
  END IF
END SUBROUTINE
