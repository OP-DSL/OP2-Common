


program airfoil

use OP2_C
use constantVars
use airfoil_seq

use, intrinsic :: ISO_C_BINDING


  implicit none

  intrinsic :: sqrt, real

  !external save_soln, adt_calc, res_calc, bres_calc, update

	! debug variables
!	integer(c_int) :: retDebug, dataSize
!!	character(kind=c_char) :: debfilename(20)
!	real(c_double) :: datad
!	integer(4) :: debugiter


  integer(4) :: iter, k, i

  integer(4), parameter :: maxnode = 9900
  integer(4), parameter :: maxcell = (9702+1)
  integer(4), parameter :: maxedge = 19502

	integer(4), parameter :: iterationNumber = 1000


  integer(4) :: nnode, ncell, nbedge, nedge, niter
  real(8) :: ncellr

  ! integer references (valid inside the OP2 library) for op_set
  type(op_set) :: nodes, edges, bedges, cells

  ! integer references (valid inside the OP2 library) for pointers between data sets
  type(op_map) :: pedge, pecell, pcell, pbedge, pbecell

  ! integer reference (valid inside the OP2 library) for op_data 
  type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res, p_rms
  
  ! arrays used in data
  integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
  real(8), dimension(:), allocatable, target :: x, q, qold, adt, res, rms

	character(kind=c_char,len=10) :: savesolnName = C_CHAR_'save_soln'//C_NULL_CHAR
	character(kind=c_char, len=9) :: adtcalcName = C_CHAR_'adt_calc' // C_NULL_CHAR
	character(kind=c_char, len=9) :: rescalcName = C_CHAR_'res_calc' // C_NULL_CHAR
	character(kind=c_char,len=10) :: brescalcName = C_CHAR_'bres_calc' // C_NULL_CHAR
	character(kind=c_char, len=7) :: updateName = C_CHAR_'update' // C_NULL_CHAR

	character(kind=c_char,len=6) :: nodesName =	C_CHAR_'nodes'//C_NULL_CHAR
	character(kind=c_char,len=6) :: edgesName =	C_CHAR_'edges'//C_NULL_CHAR
	character(kind=c_char,len=7) :: bedgesName = C_CHAR_'bedges'//C_NULL_CHAR
	character(kind=c_char,len=6) :: cellsName =	C_CHAR_'cells'//C_NULL_CHAR

	character(kind=c_char,len=6) :: pedgeName =	C_CHAR_'pedge'//C_NULL_CHAR
	character(kind=c_char,len=7) :: pecellName = C_CHAR_'pecell'//C_NULL_CHAR
	character(kind=c_char,len=6) :: pcellName =	C_CHAR_'pcell'//C_NULL_CHAR
	character(kind=c_char,len=7) :: pbedgeName =	C_CHAR_'pbedge'//C_NULL_CHAR
	character(kind=c_char,len=8) :: pbecellName =	C_CHAR_'pbecell'//C_NULL_CHAR

	character(kind=c_char,len=6) :: boundName =	C_CHAR_'bound'//C_NULL_CHAR
	character(kind=c_char,len=2) :: xName =	C_CHAR_'x'//C_NULL_CHAR
	character(kind=c_char,len=2) :: qName =	C_CHAR_'q'//C_NULL_CHAR
	character(kind=c_char,len=5) :: qoldName =	C_CHAR_'qold'//C_NULL_CHAR
	character(kind=c_char,len=4) :: adtName =	C_CHAR_'adt'//C_NULL_CHAR
	character(kind=c_char,len=4) :: resName =	C_CHAR_'res'//C_NULL_CHAR

	! read set sizes from input file (input is subdivided in two routines as we cannot allocate arrays in subroutines in
	! fortran 90)
	call getSetSizes ( nnode, ncell, nedge, nbedge )
	
	! allocate sets (cannot allocate in subroutine in F90)
	allocate ( cell ( 4 * ncell ) )
	allocate ( edge ( 2 * nedge ) )
	allocate ( ecell ( 2 * nedge ) )
	allocate ( bedge ( 2 * nbedge ) )
	allocate ( becell ( nbedge ) )
	allocate ( bound ( nbedge ) )
		
	allocate ( x ( 2 * nnode ) )
	allocate ( q ( 4 * ncell ) )
	allocate ( qold ( 4 * ncell ) )
	allocate ( res ( 4 * ncell ) )
	allocate ( adt ( ncell ) )

	allocate ( rms ( 1 ) )

	! fill up arrays from file
	call getSetInfo ( nnode, ncell, nedge, nbedge, cell, edge, ecell, bedge, becell, bound, x, q, qold, res, adt )

  ! set constants and initialise flow field and residual
	call initialise_flow_field ( ncell, q, res )
	
  do iter = 1, 4*ncell
    res(iter) = 0.0
  end do 

  ! OP initialisation
	call op_init()
	
  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
  call op_decl_set ( nnode, nodes, nodesName )
  call op_decl_set ( nedge, edges, edgesName )
	call op_decl_set ( nbedge, bedges, bedgesName )
  call op_decl_set ( ncell, cells, cellsName )

  call op_decl_map ( edges, nodes, 2, edge, pedge, pedgeName ) 
  call op_decl_map ( edges, cells, 2, ecell, pecell, pecellName )
	call op_decl_map ( bedges, nodes, 2, bedge, pbedge, pbedgeName )
	call op_decl_map ( bedges, cells, 1, becell, pbecell, pecellName )
  call op_decl_map ( cells, nodes, 4, cell, pcell, pcellName )

  call op_decl_dat ( bedges, 1, bound, p_bound, boundName )	
  call op_decl_dat ( nodes, 2, x, p_x, xName )
  call op_decl_dat ( cells, 4, q, p_q, qName )
	call op_decl_dat ( cells, 4, qold, p_qold, qoldName )
  call op_decl_dat ( cells, 1, adt, p_adt, adtName )
  call op_decl_dat ( cells, 4, res, p_res, resName )
	
	call op_decl_gbl ( rms, 1, p_rms )
	
	! main time-marching loop
	
  do niter = 1, iterationNumber

		! save old flow solution

    call op_par_loop_2 ( save_soln, cells, &
                       & p_q,    -1, OP_ID, OP_READ, &
                       & p_qold, -1, OP_ID, OP_WRITE &
                     & )
	
		! predictor/corrector update loop
				
    do k = 1, 2

      ! calculate area/timstep

      call op_par_loop_6 ( adt_calc, cells, &
                         & p_x,   1, pcell, OP_READ, &
                         & p_x,   2, pcell, OP_READ, &
                         & p_x,   3, pcell, OP_READ, &
                         & p_x,   4, pcell, OP_READ, &
                         & p_q,   -1, OP_ID, OP_READ, &
                         & p_adt, -1, OP_ID, OP_WRITE &
                       & )



      ! calculate flux residual

      call op_par_loop_8 ( res_calc, edges, &
                         & p_x,    1, pedge,  OP_READ, &
                         & p_x,    2, pedge,  OP_READ, &
                         & p_q,    1, pecell, OP_READ, &
                         & p_q,    2, pecell, OP_READ, &
                         & p_adt,  1, pecell, OP_READ, &
                         & p_adt,  2, pecell, OP_READ, &
                         & p_res,  1, pecell, OP_INC,  &
                         & p_res,  2, pecell, OP_INC   &
											 & )


      call op_par_loop_6 ( bres_calc, bedges, &
												 & p_x,      1, pbedge,  OP_READ, &
												 & p_x,      2, pbedge,  OP_READ, &
												 & p_q,      1, pbecell, OP_READ, &
												 & p_adt,    1, pbecell, OP_READ, &
											   & p_res,    1, pbecell, OP_INC,  &
                         & p_bound,  -1, OP_ID, OP_READ  &
											 & )

      ! update flow field
      
      rms(1) = 0.0

      call op_par_loop_5 ( update, cells, &
                         & p_qold, -1, OP_ID,  OP_READ,  &
                         & p_q,    -1, OP_ID,  OP_WRITE, &
                         & p_res,  -1, OP_ID,  OP_RW,    &
                         & p_adt,  -1, OP_ID,  OP_READ,  &
                         & p_rms,  -1, OP_GBL, OP_INC    &
                       & )


    end do ! internal loop

    ncellr = real ( ncell )
    rms(1) = sqrt ( rms(1) / ncellr )
	
  end do ! external loop

	! uncomment the following statements to get the result of the airfoil written in a file
!	retDebug = openfile ( C_CHAR_"/work/cbertoll/airfoil-generated/q.txt"//C_NULL_CHAR )
!
!	do debugiter = 1, 4*ncell
!
!		datad = q(debugiter)
!		retDebug = writeRealToFile ( datad )
!
!	retDebug = closefile ()
!	end do


end program airfoil