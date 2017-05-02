SUBROUTINE op_wrap_save_soln( &
    & opDat1Local, &
    & opDat2Local, &
    & bottom,top)
  implicit none
  real(8) opDat1Local(4,top)
real(8) opDat2Local(4,top)
  INTEGER(kind=4) bottom,top,i1
!$omp target teams distribute parallel do map(to:opDat1Local) map(from:opDat2Local)
  DO i1 = bottom, top-1, 1
  ! kernel call
  opDat2Local(1,i1+1) = opDat1Local(1,i1+1)
  opDat2Local(2,i1+1) = opDat1Local(2,i1+1)
  opDat2Local(3,i1+1) = opDat1Local(3,i1+1)
  opDat2Local(4,i1+1) = opDat1Local(4,i1+1)
  END DO
!$omp end target teams distribute parallel do
END SUBROUTINE

      program mytest
      implicit none
      real*4    :: B(4,10), C(4,10), sum
      integer :: N, i, j
      B(:,:) = 3.0_4
      C(:,:) = 4.0_4

      call op_wrap_save_soln(B,C,0,10)
      sum = 0.0e0
      !$omp target teams distribute parallel do reduction(+:sum) map(tofrom:sum) map(to: B,C)
       do i = 1,10
        do j = 1,4
          sum = sum + B(j,i) * C(j,i)
        end do
       end do
      !a$omp end target teams
      print *,sum
      end program mytest
