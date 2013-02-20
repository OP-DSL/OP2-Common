SUBROUTINE update(qold,q,res,adt,rms)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(*) :: qold
  REAL(kind=8), DIMENSION(*) :: q
  REAL(kind=8), DIMENSION(*) :: res
  REAL(kind=8), DIMENSION(*) :: adt
  REAL(kind=8), DIMENSION(1) :: rms
  REAL(kind=8) :: del,adti
  INTEGER(kind=4) :: i
  adti = 1.0 / adt(1)

  DO i = 1, 4
    del = adti * res(i)
    q(i) = qold(i) - del
    res(i) = 0.0
    rms(1) = rms(1) + del * del
  END DO
END SUBROUTINE
