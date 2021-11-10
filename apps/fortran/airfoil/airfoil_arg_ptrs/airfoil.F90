program AIRFOIL
  use OP2_FORTRAN_DECLARATIONS
!  use OP2_FORTRAN_HDF5_DECLARATIONS
  use OP2_Fortran_Reference
  use OP2_CONSTANTS
  use AIRFOIL_SEQ
  use IO
  use, intrinsic :: ISO_C_BINDING

  implicit none

  intrinsic :: sqrt, real

  integer(4) :: iter, k, i

  integer(4), parameter :: maxnode = 9900
  integer(4), parameter :: maxcell = (9702+1)
  integer(4), parameter :: maxedge = 19502

  integer(4), parameter :: iterationNumber = 1000

  integer(4) :: nnode, ncell, nbedge, nedge, niter, qdim
  real(8) :: ncellr

  ! profiling
  real(kind=c_double) :: startTime = 0
  real(kind=c_double) :: endTime = 0

  ! integer references (valid inside the OP2 library) for op_set
  type(op_set) :: nodes, edges, bedges, cells

  ! integer references (valid inside the OP2 library) for pointers between data sets
  type(op_map) :: pedge, pecell, pcell, pbedge, pbecell

  ! integer reference (valid inside the OP2 library) for op_data
  type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res

  ! arrays used in data
  real*8 x(1), q(1), qold(1), adt(1), res(1), q_part(1)
  pointer (ptr_x,x), (ptr_q,q), (ptr_qold, qold), (ptr_adt, adt), &
  & (ptr_res,res), (ptr_q_part, q_part)

  integer*4  ecell(1), bound(1), edge(1), bedge(1), becell(1), cell(1)
  pointer (ptr_ecell, ecell), (ptr_bound, bound), (ptr_edge, edge), &
  & (ptr_bedge, bedge), (ptr_becell, becell), (ptr_cell, cell)

  real(8), dimension(1:2) :: rms

  integer(4) :: snode, scell, sedge, sbedge
  integer(4) :: debugiter, retDebug
  real(8) :: datad

  ! for validation
  REAL(KIND=8) :: diff
  integer(4):: ncelli

  integer(4), parameter :: FILE_ID = 10

  ! read set sizes from input file (input is subdivided in two routines as we cannot allocate arrays in subroutines in
  ! fortran 90)
  print *, "Getting set sizes"
  call getSetSizes ( nnode, ncell, nedge, nbedge )
  snode = 2
  scell = 1
  sedge = 5
  sbedge = 7

  print *, ncell
  ! allocate sets (cannot allocate in subroutine in F90)
  call op_memalloc(ptr_cell, 4 * ncell  * 4)
  call op_memalloc(ptr_edge, 2 * nedge * 4)
  call op_memalloc(ptr_ecell, 2 * nedge * 4)
  call op_memalloc(ptr_bedge, 2 * nbedge * 4)
  call op_memalloc(ptr_becell,    nbedge * 4)
  call op_memalloc(ptr_bound,    nbedge * 4)

  call op_memalloc(ptr_x, 2 * nnode * 8 )
  call op_memalloc(ptr_q, 4 * ncell  * 8)
  call op_memalloc(ptr_qold, 4 * ncell  * 8)
  call op_memalloc(ptr_res, 4 * ncell  * 8)
  call op_memalloc(ptr_adt, ncell  * 8)

  call op_memalloc(ptr_q_part,  4 * ncell)

  print *, "Getting data"

  open ( FILE_ID, file = 'new_grid.dat' )

  call getSetInfo ( nnode, ncell, nedge, nbedge, cell, edge, ecell, bedge, &
  & becell, bound, x, q, qold, res, adt )

  ! OP initialisation
  call op_init_base (0,0)

  print *, "Initialising constants"
  call initialise_flow_field ( ncell, q, res )

  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
  print *, "Declaring OP2 sets"
  call op_decl_set ( nnode, nodes, 'nodes' )
  call op_decl_set ( nedge, edges, 'edges' )
  call op_decl_set ( nbedge, bedges, 'bedges' )
  call op_decl_set ( ncell, cells, 'cells' )

  call op_register_set(snode, nodes)
  call op_register_set(scell, cells)
  call op_register_set(sedge, edges)
  call op_register_set(sbedge, bedges)


  print *, "Declaring OP2 maps"
  call op_decl_map ( edges, nodes, 2, edge, pedge, 'pedge' )
  call free(ptr_edge)
  call op_decl_map ( edges, cells, 2, ecell, pecell, 'pecell' )
  call free(ptr_ecell)
  call op_decl_map ( bedges, nodes, 2, bedge, pbedge, 'pbedge' )
  call free(ptr_bedge)
  call op_decl_map ( bedges, cells, 1, becell, pbecell, 'pbecell' )
  call free(ptr_becell)
  call op_decl_map ( cells, nodes, 4, cell, pcell, 'pcell' )
  call free(ptr_cell)

  print *, "Declaring OP2 data"
  call op_decl_dat ( bedges, 1, 'integer' ,bound, p_bound, 'p_bound')
  call free(ptr_bound)
  call op_decl_dat ( nodes, 2, 'real(8)',x, p_x, 'p_x' )
  call free(ptr_x)
  call op_decl_dat ( cells, 4, 'real(8)', q, p_q, 'p_q' )
  call free(ptr_q)
  call op_decl_dat ( cells, 4, 'real(8)', qold, p_qold, 'p_qold' )
  call free(ptr_qold)
  call op_decl_dat ( cells, 1, 'real(8)', adt, p_adt, 'p_adt' )
  call free(ptr_adt)
  call op_decl_dat ( cells, 4, 'real(8)', res, p_res, 'p_res' )
  call free(ptr_res)


  print *, "Declaring OP2 constants"
  call op_decl_const(gam, 1, 'gam')
  call op_decl_const(gm1, 1, 'gm1')
  call op_decl_const(cfl, 1, 'cfl')
  call op_decl_const(eps, 1, 'eps')
  call op_decl_const(mach, 1, 'mach')
  call op_decl_const(alpha, 1, 'alpha')
  call op_decl_const(qinf, 4, 'qinf')

  !call op_dump_to_hdf5("new_grid_out.h5")
  !call op_fetch_data_hdf5_file(p_x, "new_grid_out.h5")

  call op_partition ('PTSCOTCH','KWAY', edges, pecell, p_x)

  ncelli  = op_get_size(cells)
  ncellr = real(ncelli)

  ! start timer
  call op_timers ( startTime )

  ! main time-marching loop

  do niter = 1, iterationNumber

     call op_par_loop_2 ( save_soln, op_get_set(scell), &
                       & op_arg_dat (q,    -1, OP_ID, 4,"real(8)", OP_READ), &
                       & op_arg_dat (qold, -1, OP_ID, 4,"real(8)", OP_WRITE))

    ! predictor/corrector update loop

    do k = 1, 2

      ! calculate area/timstep
      call op_par_loop_6 ( adt_calc, cells, &
                         & op_arg_dat (x,    1, cell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (x,    2, cell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (x,    3, cell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (x,    4, cell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (q,   -1, OP_ID, 4,"real(8)", OP_READ), &
                         & op_arg_dat (adt, -1, OP_ID, 1,"real(8)", OP_WRITE))

      ! calculate flux residual
      call op_par_loop_8 ( res_calc, edges, &
                         & op_arg_dat (x,    1, edge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (x,    2, edge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (q,    1, ecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (q,    2, ecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (adt,  1, ecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (adt,  2, ecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (res,  1, ecell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (res,  2, ecell, 4,"real(8)", OP_INC))

      call op_par_loop_6 ( bres_calc, bedges, &
                         & op_opt_arg_dat (.true., x,      1, bedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (x,      2, bedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (q,      1, becell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (adt,    1, becell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (res,    1, becell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (bound, -1, OP_ID, 1,"integer(4)", OP_READ))

      ! update flow field

      rms(1:2) = 0.0

      call op_par_loop_5 ( update, cells, &
                         & op_arg_dat (qold, -1, OP_ID, 4,"real(8)",  OP_READ),  &
                         & op_arg_dat (q,    -1, OP_ID, 4,"real(8)",  OP_WRITE), &
                         & op_arg_dat (res,  -1, OP_ID, 4,"real(8)",  OP_RW),    &
                         & op_arg_dat (adt,  -1, OP_ID, 1,"real(8)",  OP_READ),  &
                         & op_arg_gbl (rms, 2, "real(8)", OP_INC))

    end do ! internal loop

    rms(2) = sqrt ( rms(2) / ncellr )

    if (op_is_root() .eq. 1) then
      if (mod(niter,100) .eq. 0) then
        write (*,*) niter,"  ",rms(2)
      end if
      if ((mod(niter,1000) .eq. 0) .AND. (ncelli == 720000) ) then
        diff=ABS((100.0_8*(rms(2)/0.0001060114637578_8))-100.0_8)
        !write (*,*) niter,"  ",rms(2)
        WRITE(*,'(a,i0,a,e16.7,a)')"Test problem with ", ncelli , &
        & " cells is within ",diff,"% of the expected solution"
        if(diff.LT.0.00001) THEN
          WRITE(*,*)"This test is considered PASSED"
        else
          WRITE(*,*)"This test is considered FAILED"
        endif
      end if
    end if

  end do ! external loop

!  call op_fetch_data(p_q,q)

!  call op_fetch_data_idx(p_q,q_part, 1, ncell)

  call op_timers ( endTime )
  call op_timing_output ()
  write (*,*) 'Max total runtime =',endTime-startTime,'seconds'

end program AIRFOIL
