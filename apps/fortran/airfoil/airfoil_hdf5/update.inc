SUBROUTINE update(qold,q,res,adt,rms,maxerr, idx, errloc)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(4), INTENT(IN) :: qold
  REAL(kind=8), DIMENSION(4) :: q
  REAL(kind=8), DIMENSION(4) :: res
  REAL(kind=8), INTENT(IN) :: adt
  REAL(kind=8), DIMENSION(2) :: rms
  REAL(kind=8) :: maxerr
  INTEGER(kind=4) :: idx,errloc
  REAL(kind=8) :: del,adti
  INTEGER(kind=4) :: i
  adti = 1.0 / adt

  DO i = 1, 4
    del = adti * res(i)
    q(i) = qold(i) - del
    res(i) = 0.0
    rms(2) = rms(2) + del * del
    if (del*del .gt. maxerr) then
      maxerr = del*del
      errloc = idx
    endif
  END DO
END SUBROUTINE
