module airfoil_seq

use OP2_CONSTANTS

 contains

! save old solution
subroutine save_soln ( q, qold )

  implicit none

  ! declaration of formal parameters
  real(8), dimension(4) :: q
  real(8), dimension(4) :: qold


  ! iteration variable
  integer(4) :: i

  ! size_q and size_qold are the same value
  do i = 1, 4
    qold(i) = q(i)
  end do

end subroutine save_soln


subroutine adt_calc ( x1, x2, x3, x4, q, adt )

  implicit none

  ! formal parameters
  real(8), dimension(2) :: x1
  real(8), dimension(2) :: x2
  real(8), dimension(2) :: x3
  real(8), dimension(2) :: x4
  real(8), dimension(4) :: q
  real(8), dimension(1) :: adt

  ! local variables
  real(8) :: dx, dy, ri, u, v, c

  ! computation
  ri = 1.0 / q(1)
  u = ri * q(2)
  v = ri * q(3)
  c = sqrt ( gam * gm1 * ( ri * q(4) - 0.5 * ( u*u + v*v ) ) )

  dx = x2(1) - x1(1)
  dy = x2(2) - x1(2)
  adt(1) = abs ( u * dy - v * dx ) + c * sqrt ( dx*dx + dy*dy )

  dx = x3(1) - x2(1)
  dy = x3(2) - x2(2)
  adt(1) = adt(1) + abs ( u * dy - v * dx ) + c * sqrt ( dx*dx + dy*dy )


  dx = x4(1) - x3(1)
  dy = x4(2) - x3(2)
  adt(1) = adt(1) + abs ( u * dy - v * dx ) + c * sqrt ( dx*dx + dy*dy )

  dx = x1(1) - x4(1)
  dy = x1(2) - x4(2)
  adt(1) = adt(1) + abs ( u * dy - v * dx ) + c * sqrt ( dx*dx + dy*dy )

  adt(1) = adt(1) / cfl

end subroutine adt_calc

subroutine res_calc ( x1, x2, q1, q2, adt1, adt2, res1, res2 )

  implicit none

  ! formal parameters
  real(8), dimension(2) :: x1
  real(8), dimension(2) :: x2
  real(8), dimension(4) :: q1
  real(8), dimension(4) :: q2
  real(8), dimension(1) :: adt1
  real(8), dimension(1) :: adt2
  real(8), dimension(4) :: res1
  real(8), dimension(4) :: res2

  ! local variables
  real(8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f

  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)

  ri   = 1.0 / q1(1)
  p1   = gm1 * (q1(4)-0.5 * ri * (q1(2)*q1(2)+q1(3)*q1(3)))
  vol1 =  ri * (q1(2)*dy - q1(3)*dx)

  ri   = 1.0 / q2(1)
  p2   = gm1 * (q2(4)-0.5 * ri*(q2(2)*q2(2)+q2(3)*q2(3)))
  vol2 =  ri * (q2(2)*dy - q2(3)*dx)


  mu = 0.5 * ((adt1(1))+(adt2(1))) * eps


  f = 0.5 * (vol1 * q1(1)         + vol2* q2(1)        ) + mu*(q1(1)-q2(1))
  res1(1) = res1(1) + f
  res2(1) = res2(1) - f

  f = 0.5 * (vol1* q1(2) + p1*dy + vol2* q2(2) + p2*dy) + mu*(q1(2)-q2(2))
  res1(2) = res1(2) + f
  res2(2) = res2(2) - f

  f = 0.5 * (vol1* q1(3) - p1*dx + vol2* q2(3) - p2*dx) + mu*(q1(3)-q2(3))
  res1(3) = res1(3) + f
  res2(3) = res2(3) - f

  f = 0.5 * (vol1*(q1(4)+p1)     + vol2*(q2(4)+p2)    ) + mu*(q1(4)-q2(4))
  res1(4) = res1(4) + f
  res2(4) = res2(4) - f

end subroutine res_calc


subroutine bres_calc ( x1, x2, q1, adt1, res1, bound )

  implicit none

  ! formal parameters
  real(8), dimension(*) :: x1
  real(8), dimension(*) :: x2
  real(8), dimension(*)  :: q1
  real(8), dimension(*) :: adt1
  real(8), dimension(*) :: res1
  integer(4), dimension(*) :: bound

  ! local variables
  real(8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f

  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)

  ri = 1.0 / q1(1)
  p1 = gm1*(q1(4)-0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)));


  if ( bound(1) == 1 ) then

    res1(2) = res1(2) + (+ p1*dy)
    res1(3) = res1(3) + (- p1*dx)

  else

    vol1 =  ri*(q1(2)*dy - q1(3)*dx)

    ri   = 1.0 / qinf(1);
    p2   = gm1*(qinf(4)-0.5 * ri * (qinf(2)*qinf(2)+qinf(3)*qinf(3)));
    vol2 =  ri*(qinf(2)*dy - qinf(3)*dx);

    mu = (adt1(1)) * eps;


    f = 0.5 * (vol1 * q1(1)         + vol2 * qinf(1)        ) + mu*(q1(1)-qinf(1));
    res1(1) = res1(1) + f;


    f = 0.5 * (vol1* q1(2) + p1*dy + vol2* qinf(2) + p2*dy) + mu*(q1(2)-qinf(2));
    res1(2) = res1(2) + f;


    f = 0.5 * (vol1* q1(3) - p1*dx + vol2* qinf(3) - p2*dx) + mu*(q1(3)-qinf(3));
    res1(3) = res1(3) + f;


    f = 0.5 * (vol1*(q1(4)+p1)     + vol2*(qinf(4)+p2)    ) + mu*(q1(4)-qinf(4));
    res1(4) = res1(4) + f;

  end if

end subroutine bres_calc


subroutine update ( qold, q, res, adt, rms )

  implicit none

  ! formal parameters
  real(8), dimension(*) :: qold
  real(8), dimension(*) :: q
  real(8), dimension(*) :: res
  real(8), dimension(*) :: adt
  real(8), dimension(1) :: rms

  real(8) :: del, adti, debug

  integer(4) :: i

  adti = 1.0 / adt(1)

  do i = 1, 4
    del = adti * res(i)
    q(i) = qold(i) - del
    res(i) = 0.0
    rms(1) = rms(1) + del * del
  end do


end subroutine update

end subroutine initialise_flow_field

end module airfoil_seq
