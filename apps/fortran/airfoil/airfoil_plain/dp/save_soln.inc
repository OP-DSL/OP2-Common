SUBROUTINE save_soln(q,qold)
  IMPLICIT NONE

  REAL(kind=8), DIMENSION(4), INTENT(IN) :: q
  REAL(kind=8), DIMENSION(4) :: qold
  INTEGER(kind=4) :: i

  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE
