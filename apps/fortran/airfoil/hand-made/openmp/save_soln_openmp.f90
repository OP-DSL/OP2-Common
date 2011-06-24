
module save_soln_openmp

#ifdef _OPENMP
	use omp_lib
#endif

  use OP2_Fortran_Declarations
  use OP2_Fortran_RT_Support
	use airfoil_seq
	use OP2Profiling

	real(8), dimension(:), pointer :: argument1
	real(8), dimension(:), pointer :: argument2

	logical :: isFirstTimeExecuting_save_soln = .true.

	contains
	
		subroutine save_soln_caller (	callerArgument1, &
															  & callerArgument2, &
															  & sliceStart, sliceEnd &
														  & )
			real(8), dimension(0:*) :: callerArgument1
			real(8), dimension(0:*) :: callerArgument2
			
			integer(kind = OMP_integer_kind) :: sliceStart
			integer(kind = OMP_integer_kind) :: sliceEnd

			integer(kind = OMP_integer_kind) :: sliceIterator

			integer(kind = OMP_integer_kind) :: threadid
			
			threadid = OMP_get_thread_num() 

			! apply the kernel to each element of the assigned slice of set
			do sliceIterator = sliceStart, sliceEnd-1

				call save_soln ( callerArgument1 ( sliceIterator * 4: sliceIterator * 4 + 4 - 1 ), &
											 & callerArgument2 ( sliceIterator * 4: sliceIterator * 4 + 4 - 1 ) &
										 & )
			end do
			

		
		end subroutine save_soln_caller

	
		function op_par_loop_save_soln ( subroutineName, setIn, &
																	 & arg1In, idx1, map1In, access1, &
																	 & arg2In, idx2, map2In, access2 &
																 & )

			! use directives	
			use, intrinsic :: ISO_C_BINDING

			! mandatory	
			implicit none

			type(profInfo) :: op_par_loop_save_soln
			
			! formal arguments
			character, dimension(*), intent(in) :: subroutineName
			
			! data set on which we loop
			type(op_set), intent(in) :: setIn

			! data ids used in the function
			type(op_dat) :: arg1In, arg2In
			
			! index to be used in first and second pointers
			integer(4), intent(in) :: idx1, idx2
			
			! ptr ids for indirect access to data
			type(op_map) :: map1In, map2In
			
			! access values for arguments
			integer(4), intent(in) :: access1, access2

			! local variables
      type(op_set_core), pointer :: set
      type(op_map_core), pointer :: map1, map2
      type(op_dat_core), pointer :: arg1, arg2


			! number of threads
			integer(kind = OMP_integer_kind) :: nthreads = 0

			! thread index
			integer(kind = OMP_integer_kind) :: threadIndex = -1

			! bounds of set slice assigned to each thread
			integer(kind = OMP_integer_kind) :: sliceStart = -1
			integer(kind = OMP_integer_kind) :: sliceEnd = -1

			! initialise timers
      real(kind=c_double) :: elapsedTimeStart = 0
      real(kind=c_double) :: elapsedTimeEnd = 0
			
      type(op_set_core), pointer :: arg1Set, arg2Set

      integer(4) :: data1Size, data2Size

			! get number of threads
#ifdef _OPENMP
			nthreads = omp_get_max_threads ()
#else
			nthreads = 1
#endif

      ! initialise input data
      set => setIn%setPtr
      
      map1 => map1In%mapPtr
      map2 => map2In%mapPtr

      arg1 => arg1In%dataPtr
      arg2 => arg2In%dataPtr

			if ( isFirstTimeExecuting_save_soln .eqv. .true. ) then

        call c_f_pointer ( arg1%set, arg1Set )
        data1Size = ( arg1%dim * arg1Set%size)

        call c_f_pointer ( arg2%set, arg2Set )
        data2Size = ( arg2%dim * arg2Set%size)
			
				call c_f_pointer ( arg1%dat, argument1, (/data1Size/) )
				call c_f_pointer ( arg2%dat, argument2, (/data2Size/) )
			
				isFirstTimeExecuting_save_soln = .false.
			
			end if
			
      ! start time
			call op_timers ( elapsedTimeStart )

			
			! apply kernel to each set element (no plan in direct loops)
			
			! private is required in fortran because the declaration is global, and we can't declare local thread
			! variables inside the do below
			
			! notice also that the DO pragma, without PARALLEL, does not include a synchronisation
			
			!$OMP PARALLEL DO PRIVATE(sliceStart,sliceEnd)
			do threadIndex = 0, nthreads-1
			
				sliceStart = ( set%size * (threadIndex) ) / nthreads;
				sliceEnd = ( set%size * ( (threadIndex) + 1 ) ) / nthreads;
				
				call save_soln_caller ( argument1, &
															& argument2, &
															& sliceStart, sliceEnd &
														& )
				
			end do
			!$OMP END PARALLEL DO
						
      call op_timers ( elapsedTimeEnd )
						
			! timing end
			op_par_loop_save_soln%hostTime = elapsedTimeEnd - elapsedTimeStart
			op_par_loop_save_soln%kernelTime = 0.0

		end function op_par_loop_save_soln

end module save_soln_openmp