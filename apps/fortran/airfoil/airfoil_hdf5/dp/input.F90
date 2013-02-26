module IO
#ifdef OP2_WITH_CUDAFOR
  USE CUDAFOR
#endif
  USE OP2_CONSTANTS
  IMPLICIT  NONE
  contains
! read set sizes from input
subroutine getSetSizes ( nnode, ncell, nedge, nbedge )

  implicit none

  integer(4), parameter :: MAX_PWD_LEN = 255

  ! formal parameters
  integer(4), intent (out) :: nnode, ncell, nedge, nbedge

  ! file identifier (10 is arbitrary)
  integer(4), parameter :: FILE_ID = 10


  character(len=MAX_PWD_LEN) :: currDir
  call get_environment_variable ( "WORK", currDir )

  currDir = trim(currDir) //  'new_grid.dat'
  ! iterator for file scanning and array addressing

  ! open file

  open ( FILE_ID, file = currDir )

  ! first line includes number of cells, nodes, edges and bedges
  read ( FILE_ID, "(1x,I6,1x,I6,1x,I7,1x,I4)" ) nnode, ncell, nedge, nbedge

  ! not closing file because it will be used below

end subroutine getSetSizes


! fill up arrays from file
subroutine getSetInfo ( nnode, ncell, nedge, nbedge, cell, edge, ecell, bedge, becell, bound, x, q, qold, res, adt )

  implicit none

  ! formal parameters

  integer(4), intent (in) :: nnode, ncell, nedge, nbedge

  integer(4), dimension( 4 * ncell ) :: cell
  integer(4), dimension( 2 * nedge ) :: edge
  integer(4), dimension( 2 * nedge ) :: ecell
  integer(4), dimension( 2 * nbedge ) :: bedge
  integer(4), dimension( nbedge ) :: becell
  integer(4), dimension( nbedge ) :: bound

  real(8), dimension( 2 * nnode ) :: x
  real(8), dimension( 4 * ncell ) :: q
  real(8), dimension( 4 * ncell ) :: qold
  real(8), dimension( 4 * ncell ) :: res
  real(8), dimension( ncell ) :: adt

  ! file identifier (10 is arbitrary)
  integer(4), parameter :: FILE_ID = 10

  integer(4) :: i, f_array

  ! the file is already open and the pointer in file is already at the correct position

  do i = 1, nnode
    read ( FILE_ID, * ) x( (2*i) - 1 ), x( 2*i )
  end do

  do i = 1, ncell
    read ( FILE_ID, * ) cell(4 * i - 3), cell(4 * i + 1 - 3), cell(4 * i + 2 - 3), cell(4 * i + 3 - 3 )

    ! pointers are expressed for C arrays (from 0 to N-1), for Fortran we have to convert them in sets from 1 to N
!		cell(4 * i - 3) = cell(4 * i - 3) + 1
!		cell(4 * i + 1 - 3) = cell(4 * i + 1 - 3) + 1
!		cell(4 * i + 2 - 3) = cell(4 * i + 2 - 3) + 1
!		cell(4 * i + 3 - 3 ) = cell(4 * i + 3 - 3 ) + 1
  enddo

  do i = 1, nedge
    read ( FILE_ID, * ) edge(2 * i - 1), edge(2 * i + 1 - 1), ecell(2 * i - 1), ecell(2 * i + 1 - 1)

    ! see above
!		edge(2 * i - 1) = edge(2 * i - 1) + 1
!		edge(2 * i + 1 - 1) = edge(2 * i + 1 - 1) + 1
!		ecell(2 * i - 1) = ecell(2 * i - 1) + 1
!		ecell(2 * i + 1 - 1) = ecell(2 * i + 1 - 1) + 1
  enddo

  do i = 1, nbedge
    read ( FILE_ID, * ) bedge(2 * i - 1), bedge(2 * i + 1 - 1), becell(i), bound(i)

    ! see above
!		bedge(2 * i - 1) = bedge(2 * i - 1) + 1
!		bedge(2 * i + 1 - 1) = bedge(2 * i + 1 - 1) + 1
!		becell(i) = becell(i) + 1
  enddo

  ! close file
  close ( FILE_ID )


end subroutine getSetInfo

subroutine initialise_flow_field ( ncell, q, res )

  ! formal parameters
  integer(4) :: ncell

  real(8), dimension(:) :: q
  real(8), dimension(:) :: res

  ! local variables
  real(8) :: p, r, u, e

  integer(4) :: n, m

  gam = 1.4
  gm1 = 1.4 - 1.0
  cfl = 0.9
  eps = 0.05

  mach  = 0.4
  alpha = 3.0 * atan(1.0) / 45.0
  p     = 1.0
  r     = 1.0
  u     = sqrt ( gam * p / r ) * mach
  e     = p / ( r * gm1 ) + 0.5 * u * u

  qinf(1) = r
  qinf(2) = r * u
  qinf(3) = 0.0
  qinf(4) = r * e

  ! -4 in the subscript is done to adapt C++ code to fortran one
  do n = 1, ncell
    do m = 1, 4
      q( (4 * n + m) - 4) = qinf(m)
      res( (4 * n + m) - 4) = 0.0
    end do
  end do

end subroutine initialise_flow_field

subroutine initialise_constants ( )

  ! local variables
  real(8) :: p, r, u, e

  gam = 1.4
  gm1 = 1.4 - 1.0
  cfl = 0.9
  eps = 0.05

  mach  = 0.4
  alpha = 3.0 * atan(1.0) / 45.0
  p     = 1.0
  r     = 1.0
  u     = sqrt ( gam * p / r ) * mach
  e     = p / ( r * gm1 ) + 0.5 * u * u

  qinf(1) = r
  qinf(2) = r * u
  qinf(3) = 0.0
  qinf(4) = r * e

end subroutine initialise_constants

end module
