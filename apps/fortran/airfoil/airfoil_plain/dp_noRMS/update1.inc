SUBROUTINE update1(qold,q,res,adt,rms)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(4), INTENT(IN) :: qold
  REAL(kind=8), DIMENSION(4) :: q
  REAL(kind=8), DIMENSION(4) :: res
  REAL(kind=8), DIMENSION(4) :: del
  REAL(kind=8), INTENT(IN) :: adt
  REAL(kind=8), DIMENSION(2) :: rms
  REAL(kind=8) :: adti
  INTEGER(kind=4) :: i
  adti = 1.0 / adt

  DO i = 1, 4
    del(i) = qold(i) - q(i)
  END DO

  DO i = 1, 4
    rms(2) = rms(2) + del(i) * del(i)
  END DO

END SUBROUTINE
