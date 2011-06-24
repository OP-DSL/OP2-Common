module update_openmp

#ifdef _OPENMP
  use omp_lib
#endif

  use OP2_Fortran_Declarations
  use OP2_Fortran_RT_Support
  use airfoil_seq
  use OP2Profiling

  real(8), dimension(:), pointer :: argument1
  real(8), dimension(:), pointer :: argument2
  real(8), dimension(:), pointer :: argument3
  real(8), dimension(:), pointer :: argument4
  real(8), dimension(:), pointer :: argument5

  integer(4) :: arg1Size, arg2Size, arg3Size, arg4Size, arg5Size

  logical :: isFirstTimeExecuting_update = .true.

contains

  subroutine update_caller ( callerArgument1, &
                           & callerArgument2, &
                           & callerArgument3, &
                           & callerArgument4, &
                           & callerArgument5, &
                           & sliceStart, sliceEnd &
                         & )

    real(8), dimension(0:*) :: callerArgument1
    real(8), dimension(0:*) :: callerArgument2
    real(8), dimension(0:*) :: callerArgument3
    real(8), dimension(0:*) :: callerArgument4
    real(8), dimension(0:*) :: callerArgument5

    integer(kind = OMP_integer_kind) :: sliceStart
    integer(kind = OMP_integer_kind) :: sliceEnd

    integer(kind = OMP_integer_kind) :: sliceIterator

    ! apply the kernel to each element of the assigned slice of set
    do sliceIterator = sliceStart, sliceEnd-1

      call update ( callerArgument1 ( sliceIterator * 4: sliceIterator * 4 + 4 - 1 ), &
                  & callerArgument2 ( sliceIterator * 4: sliceIterator * 4 + 4 - 1 ), &
                  & callerArgument3 ( sliceIterator * 4: sliceIterator * 4 + 4 - 1 ), &
                  & callerArgument4 ( sliceIterator * 1: sliceIterator * 1 + 1 - 1 ), &
                  & callerArgument5 &
                & )
    end do

  end subroutine update_caller


  function op_par_loop_update ( subroutineName, setIn, &
                              & arg1In, idx1, map1In, access1, &
                              & arg2In, idx2, map2In, access2, &
                              & arg3In, idx3, map3In, access3, &
                              & arg4In, idx4, map4In, access4, &
                              & arg5In, idx5, map5In, access5 &
                            & )

    ! use directives
    use, intrinsic :: ISO_C_BINDING

    ! mandatory
    implicit none

    type(profInfo) :: op_par_loop_update

    ! formal arguments
    character, dimension(*), intent(in) :: subroutineName

    ! data set on which we loop
    type(op_set), intent(in) :: setIn

    ! data ids used in the function
    type(op_dat) :: arg1In, arg2In, arg3In, arg4In, arg5In

    ! index to be used in first and second pointers
    integer(4), intent(in) :: idx1, idx2, idx3, idx4, idx5

    ! ptr ids for indirect access to data
    type(op_map) :: map1In, map2In, map3In, map4In, map5In

    ! access values for arguments
    integer(4), intent(in) :: access1, access2, access3, access4, access5

    ! local variables
    type(op_set_core), pointer :: set
    type(op_map_core), pointer :: map1, map2, map3, map4, map5
    type(op_dat_core), pointer :: arg1, arg2, arg3, arg4, arg5

    ! number of threads
    integer(kind = OMP_integer_kind) :: nthreads = 0

    ! thread index
    integer(kind = OMP_integer_kind) :: threadIndex = -1

    ! bounds of set slice assigned to each thread
    integer(kind = OMP_integer_kind) :: sliceStart = -1
    integer(kind = OMP_integer_kind) :: sliceEnd = -1

    real(8), dimension ( 0:(1 + 64 * 64) -1 ) :: arg5_l

    integer(4) :: iter1, iter2

    ! initialise timers
    real(kind=c_double) :: elapsedTimeStart = 0
    real(kind=c_double) :: elapsedTimeEnd = 0

    type(op_set_core), pointer :: arg1Set, arg2Set, arg3Set, arg4Set

    ! start time
    call op_timers ( elapsedTimeStart )

    ! get number of threads
#ifdef _OPENMP
    nthreads = omp_get_max_threads ()
#else
    nthreads = 1
#endif

    set => setIn%setPtr

    map1 => map1In%mapPtr
    map2 => map2In%mapPtr
    map3 => map3In%mapPtr
    map4 => map4In%mapPtr
    map5 => map5In%mapPtr

    arg1 => arg1In%dataPtr
    arg2 => arg2In%dataPtr
    arg3 => arg3In%dataPtr
    arg4 => arg4In%dataPtr
    arg5 => arg5In%dataPtr

    call c_f_pointer ( arg1%set, arg1Set )
    arg1Size = arg1%dim * arg1set%size

    call c_f_pointer ( arg2%set, arg2Set )
    arg2Size = arg2%dim * arg2set%size

    call c_f_pointer ( arg3%set, arg3Set )
    arg3Size = arg3%dim * arg3set%size

    call c_f_pointer ( arg4%set, arg4Set )
    arg4Size = arg4%dim * arg4set%size

    ! warning: the argument is a global op_dat so the size is directly the dim of the op_dat
    arg5Size = arg5%dim

    if ( isFirstTimeExecuting_update .eqv. .true. ) then

      call c_f_pointer ( arg1%dat, argument1, (/arg1Size/) )
      call c_f_pointer ( arg2%dat, argument2, (/arg2Size/) )
      call c_f_pointer ( arg3%dat, argument3, (/arg3Size/) )
      call c_f_pointer ( arg4%dat, argument4, (/arg4Size/) )
      call c_f_pointer ( arg5%dat, argument5, (/arg5Size/) )

      isFirstTimeExecuting_update = .false.

    end if

    ! apply kernel to each set element (no plan in direct loops)

    ! private is required in fortran because the declaration is global, and we can't declare local thread
    ! variables inside the do below

    ! notice also that the DO pragma, without PARALLEL, does not include a synchronisation

    do iter1 = 0, nthreads-1
      do iter2 = 0, 1 - 1

        arg5_l ( iter2 + iter1 * 64 ) = 0

      end do
    end do


    !$OMP PARALLEL DO PRIVATE(sliceStart,sliceEnd)
    do threadIndex = 0, nthreads-1

      sliceStart = ( set%size * (threadIndex) ) / nthreads;
      sliceEnd = ( set%size * ( (threadIndex) + 1 ) ) / nthreads;

      call update_caller ( argument1, &
                         & argument2, &
                         & argument3, &
                         & argument4, &
                         & arg5_l ( threadIndex * 64: ), &
                         & sliceStart, sliceEnd &
                       & )

    end do
    !$OMP END PARALLEL DO

    ! combine reduction data
    do iter1 = 0, nthreads - 1
      do iter2 = 0, 1 - 1

          argument5 ( 1 + iter2 ) = argument5 ( 1 + iter2 ) + arg5_l ( iter2 + iter1 * 64 )

      end do
    end do

    call op_timers ( elapsedTimeEnd )

    ! timing end
    op_par_loop_update%hostTime = elapsedTimeEnd - elapsedTimeStart

  end function op_par_loop_update

end module update_openmp
