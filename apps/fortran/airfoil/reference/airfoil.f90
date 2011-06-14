


program airfoil

use, intrinsic :: ISO_C_BINDING

use OP2_Fortran_Declarations
use OP2_Fortran_Reference

use constantVars
use airfoil_seq

  implicit none

  intrinsic :: sqrt, real

  integer(4) :: iter, k, i

	! debug variables
	integer(c_int) :: retDebug, dataSize
	character(kind=c_char) :: debfilename(20)
	real(c_double) :: datad
	integer(4) :: debugiter
	

  integer(4), parameter :: maxnode = 9900
  integer(4), parameter :: maxcell = (9702+1)
  integer(4), parameter :: maxedge = 19502

	integer(4), parameter :: iterationNumber = 1000


  integer(4) :: nnode, ncell, nbedge, nedge, niter
  real(8) :: ncellr


	! profiling
	real :: startTime, endTime

	type(c_funptr) :: save_soln_cptr

  ! integer references (valid inside the OP2 library) for op_set
  type(op_set) :: nodes, edges, bedges, cells

  ! integer references (valid inside the OP2 library) for pointers between data sets
  type(op_map) :: pedge, pecell, pcell, pbedge, pbecell

  ! integer reference (valid inside the OP2 library) for op_data 
  type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res, p_rms
  
  ! arrays used in data
  integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
  real(8), dimension(:), allocatable, target :: x, q, qold, adt, res, rms
		
	type(c_ptr) :: c_edge, c_ecell, c_bedge, c_becell, c_cell, c_bound, c_x, c_q, c_qold, c_adt, c_res, c_rms

	! names of sets, maps and dats
!	character(kind=c_char,len=7) ::	fakeName = C_CHAR_'NONAME'//C_NULL_CHAR

	character(len=5) :: nodesName = 'nodes'	
	character(len=5) :: edgesName = 'edges'
	character(len=6) :: bedgesName = 'bedges'
	character(len=5) :: cellsName = 'cells'
	character(len=5) :: pedgeName = 'pedge'
	character(len=6) :: pecellName = 'pecell'
	character(len=5) :: pcellName = 'pcell'
	character(len=6) :: pbedgeName = 'pbedge'
	character(len=7) :: pbecellName = 'pbecell'
	character(len=5) :: boundName = 'bound'
	character(len=1) :: xName = 'x'
	character(len=1) :: qName = 'q'
	character(len=4) :: qoldName = 'qold'
	character(len=3) :: adtName = 'adt'
	character(len=3) :: resName = 'res'
	
	integer(4) :: diags = 7
	
	! read set sizes from input file (input is subdivided in two routines as we cannot allocate arrays in subroutines in
	! fortran 90)
	call getSetSizes ( nnode, ncell, nedge, nbedge )
	
	! allocate sets (cannot allocate variables in a subroutine in F90)
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
	call op_init ( diags )

  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
  call op_decl_set ( nnode, nodes, nodesName )
  call op_decl_set ( nedge, edges, edgesName )
	call op_decl_set ( nbedge, bedges, bedgesName )
  call op_decl_set ( ncell, cells, cellsName  )

	call op_decl_map ( edges, nodes, 2, edge, pedge, pedgeName ) 
  call op_decl_map ( edges, cells, 2, ecell, pecell, pecellName )
	call op_decl_map ( bedges, nodes, 2, bedge, pbedge, pbedgeName )
	call op_decl_map ( bedges, cells, 1, becell, pbecell, pbecellName )
  call op_decl_map ( cells, nodes, 4, cell, pcell, pcellName )
	
	call op_decl_dat ( bedges, 1, bound, p_bound, boundName )	
  call op_decl_dat ( nodes, 2, x, p_x, xName )
  call op_decl_dat ( cells, 4, q, p_q, qName )
	call op_decl_dat ( cells, 4, qold, p_qold, qoldName )
	call op_decl_dat ( cells, 1, adt, p_adt, adtName )
	call op_decl_dat ( cells, 4, res, p_res, resName )
	
	call op_decl_gbl ( rms, p_rms, 1 )
				
	call op_decl_const ( 1, gam )
	call op_decl_const ( 1, gm1 )
	call op_decl_const ( 1, cfl )
	call op_decl_const ( 1, eps )
	call op_decl_const ( 1, alpha )
	call op_decl_const ( 1, air_const )
	call op_decl_const ( 4, qinf )

	! start timer: uncomment to get execution time (also uncomment cpu_time call at the end of this file)
	! call cpu_time ( startTime )
	
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

  ! uncomment to get execution time
	! call cpu_time ( endTime )
  ! write (*,*), 'Time elapsed is ', endTime - startTime, ' seconds'

! DEBUG: the following set of commands write the Q array (the actual result) to the file name below
! Change the file name to a proper absolute path.
!
! Uncomment to obtain the result
!
  retdebug = openfile ( c_char_"/data/carlo/q-seq.txt"//c_null_char )
 
  do debugiter = 1, 4*ncell
 
    datad = q(debugiter)
    retdebug = writerealtofile ( datad )
  end do
 
  retdebug = closefile ()

end program airfoil
