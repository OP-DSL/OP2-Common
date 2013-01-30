SUBROUTINE save_soln(q,qold)
  IMPLICIT NONE

  REAL(kind=8), DIMENSION(4) :: q
  REAL(kind=8), DIMENSION(*) :: qold
  INTEGER(kind=4) :: i

  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE
