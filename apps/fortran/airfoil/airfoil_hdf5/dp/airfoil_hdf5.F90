program AIRFOIL
  use OP2_FORTRAN_DECLARATIONS
  use OP2_FORTRAN_HDF5_DECLARATIONS
  use OP2_Fortran_Reference
  use OP2_Fortran_RT_Support
  use AIRFOIL_SEQ
  use OP2_CONSTANTS
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
  integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
  real(8) :: rms

  character(kind=c_char,len=6) :: nodesName  = C_CHAR_'nodes'//C_NULL_CHAR
  character(kind=c_char,len=6) :: edgesName  = C_CHAR_'edges'//C_NULL_CHAR
  character(kind=c_char,len=7) :: bedgesName = C_CHAR_'bedges'//C_NULL_CHAR
  character(kind=c_char,len=6) :: cellsName  = C_CHAR_'cells'//C_NULL_CHAR

  character(kind=c_char,len=6) :: pedgeName   = C_CHAR_'pedge'//C_NULL_CHAR
  character(kind=c_char,len=7) :: pecellName  = C_CHAR_'pecell'//C_NULL_CHAR
  character(kind=c_char,len=6) :: pcellName   = C_CHAR_'pcell'//C_NULL_CHAR
  character(kind=c_char,len=7) :: pbedgeName  = C_CHAR_'pbedge'//C_NULL_CHAR
  character(kind=c_char,len=8) :: pbecellName = C_CHAR_'pbecell'//C_NULL_CHAR

  character(kind=c_char,len=8) :: boundName = C_CHAR_'p_bound'//C_NULL_CHAR
  character(kind=c_char,len=4) :: xName     = C_CHAR_'p_x'//C_NULL_CHAR
  character(kind=c_char,len=4) :: qName     = C_CHAR_'p_q'//C_NULL_CHAR
  character(kind=c_char,len=7) :: qoldName  = C_CHAR_'p_qold'//C_NULL_CHAR
  character(kind=c_char,len=6) :: adtName   = C_CHAR_'p_adt'//C_NULL_CHAR
  character(kind=c_char,len=6) :: resName   = C_CHAR_'p_res'//C_NULL_CHAR

  character(kind=c_char,len=4) :: gamName   = C_CHAR_'gam'//C_NULL_CHAR
  character(kind=c_char,len=4) :: gm1Name   = C_CHAR_'gm1'//C_NULL_CHAR
  character(kind=c_char,len=4) :: cflName   = C_CHAR_'cfl'//C_NULL_CHAR
  character(kind=c_char,len=4) :: epsName   = C_CHAR_'eps'//C_NULL_CHAR
  character(kind=c_char,len=5) :: machName   = C_CHAR_'mach'//C_NULL_CHAR
  character(kind=c_char,len=6) :: alphaName   = C_CHAR_'alpha'//C_NULL_CHAR
  character(kind=c_char,len=5) :: qinfName   = C_CHAR_'qinf'//C_NULL_CHAR

  integer(4) :: debugiter, retDebug
  real(8) :: datad

  character(kind=c_char,len=12) :: FileName = C_CHAR_'new_grid.h5'//C_NULL_CHAR

  ! OP initialisation
  print *, "Initialising OP2"
  call op_init (0)

  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
  print *, "Declaring OP2 sets"
  call op_decl_set_hdf5 ( nnode, nodes, FileName, nodesName )
  call op_decl_set_hdf5 ( nedge, edges, FileName, edgesName )
  call op_decl_set_hdf5 ( nbedge, bedges, FileName, bedgesName )
  call op_decl_set_hdf5 ( ncell, cells, FileName, cellsName )

  print *, "Declaring OP2 maps"
  call op_decl_map_hdf5 ( edges, nodes, 2, pedge, FileName, pedgeName )
  call op_decl_map_hdf5 ( edges, cells, 2, pecell, FileName, pecellName )
  call op_decl_map_hdf5 ( bedges, nodes, 2, pbedge, FileName, pbedgeName )
  call op_decl_map_hdf5 ( bedges, cells, 1, pbecell, FileName, pbecellName )
  call op_decl_map_hdf5 ( cells, nodes, 4, pcell, FileName, pcellName )

  print *, "Declaring OP2 data"
  call op_decl_dat_hdf5 ( bedges, 1, p_bound, C_CHAR_'int'//C_NULL_CHAR, FileName, boundName )
  call op_decl_dat_hdf5 ( nodes, 2, p_x, C_CHAR_'double'//C_NULL_CHAR, FileName, xName )
  call op_decl_dat_hdf5 ( cells, 4, p_q, C_CHAR_'double'//C_NULL_CHAR, FileName, qName )
  call op_decl_dat_hdf5 ( cells, 4, p_qold, C_CHAR_'double'//C_NULL_CHAR, FileName, qoldName )
  call op_decl_dat_hdf5 ( cells, 1, p_adt, C_CHAR_'double'//C_NULL_CHAR, FileName, adtName )
  call op_decl_dat_hdf5 ( cells, 4, p_res, C_CHAR_'double'//C_NULL_CHAR, FileName, resName )

  print *, "Declaring OP2 constants"
  call op_decl_const(gam, 1, gamName)
  call op_decl_const(gm1, 1, gm1Name)
  call op_decl_const(cfl, 1, cflName)
  call op_decl_const(eps, 1, epsName)
  call op_decl_const(mach, 1, machName)
  call op_decl_const(alpha, 1, alphaName)
  call op_decl_const(qinf, 4, qinfName)

  print *, "Initialising constants"
  call initialise_constants ( )

  call op_partition (C_CHAR_'PARMETIS'//C_NULL_CHAR, C_CHAR_'KWAY'//C_NULL_CHAR, edges, pecell, p_x)
  ncellr = real(op_get_size(cells))
  print *, "ncellr", ncellr

  ! start timer
  call op_timers ( startTime )

  ! main time-marching loop

  do niter = 1, iterationNumber

     call op_par_loop_2 ( save_soln, cells, &
                       & op_arg_dat (p_q,    -1, OP_ID, 4,"real(8)", OP_READ), &
                       & op_arg_dat (p_qold, -1, OP_ID, 4,"real(8)", OP_WRITE))

    ! predictor/corrector update loop

    do k = 1, 2

      ! calculate area/timstep
      call op_par_loop_6 ( adt_calc, cells, &
                         & op_arg_dat (p_x,    1, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    2, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    3, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    4, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_q,   -1, OP_ID, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt, -1, OP_ID, 1,"real(8)", OP_WRITE))

      ! calculate flux residual
      call op_par_loop_8 ( res_calc, edges, &
                         & op_arg_dat (p_x,    1, pedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_x,    2, pedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_q,    1, pecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_q,    2, pecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,  1, pecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,  2, pecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_res,  1, pecell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (p_res,  2, pecell, 4,"real(8)", OP_INC))

      call op_par_loop_6 ( bres_calc, bedges, &
                         & op_arg_dat (p_x,      1, pbedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_x,      2, pbedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_q,      1, pbecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,    1, pbecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_res,    1, pbecell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (p_bound, -1, OP_ID, 1,"integer(4)", OP_READ))

      ! update flow field

      rms = 0.0

      call op_par_loop_5 ( update, cells, &
                         & op_arg_dat (p_qold, -1, OP_ID, 4,"real(8)",  OP_READ),  &
                         & op_arg_dat (p_q,    -1, OP_ID, 4,"real(8)",  OP_WRITE), &
                         & op_arg_dat (p_res,  -1, OP_ID, 4,"real(8)",  OP_RW),    &
                         & op_arg_dat (p_adt,  -1, OP_ID, 1,"real(8)",  OP_READ),  &
                         & op_arg_gbl (rms, 1, "real(8)", OP_INC))


    end do ! internal loop

    rms = sqrt ( rms / ncellr )

    if (mod(niter,100) .eq. 0) then
      if (op_is_root() .eq. 1) then
        write (*,*), niter,"  ",rms
      end if
    end if

  end do ! external loop

  call op_timers ( endTime )
  if (op_is_root() .eq. 1) then
    write (*,*), 'Max total runtime =', endTime - startTime,'seconds'
  end if

  call op_exit (  )
end program AIRFOIL
