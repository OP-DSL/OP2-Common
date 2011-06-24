

module adt_calc_openmp


#ifdef _OPENMP
	use omp_lib
#endif

	use OP2_Fortran_Declarations
  use OP2_Fortran_RT_Support
	use airfoil_seq
	use OP2Profiling

	use, intrinsic :: ISO_C_BINDING

	! returned plan address by cplan and plan functions
	type(c_ptr) :: planRet

	! variable for storing the actual OP Plan
	type(op_plan), pointer :: actualPlan
		
	! iteration and offset variables to implement plan execution
	integer(4) :: blockOffset, col

	! Fortran variable for Host variable
	integer, pointer, dimension(:) :: ncolblk		
	integer, pointer, dimension(:) :: pnindirect
			
	! ind_maps is an array of device pointers allocated on the host memory
	type(c_ptr), pointer, dimension(:) :: pindMaps
	integer(4), dimension(1) :: pindMapsSize ! dimension = indsNumber (= 1, see below)

	! as many pindMapsI as the positions of pindMaps (just one for this case)
	integer, dimension(:), pointer :: pindMaps1
	integer(4) :: pindMaps1Size
		
	! maps is an array of device pointers allocated on the host memory
	type(c_ptr), pointer, dimension(:) :: pmaps
	integer(4), dimension(6) :: pmapsSize ! dimension = argsNumber (= 6, see below)

	! as many pMapsI as the positions of pmaps (6 for this case) (they are short integers)
	integer(2), dimension(:), pointer :: pMaps1
	integer(2), dimension(:), pointer :: pMaps2
	integer(2), pointer, dimension(:) :: pMaps3
	integer(2), pointer, dimension(:) :: pMaps4
	integer(2), pointer, dimension(:) :: pMaps5 ! not actually used
	integer(2), pointer, dimension(:) :: pMaps6 ! not actually used

	! last two not actually used
	integer(4) :: pmaps1Size, pmaps2Size, pmaps3Size, pmaps4Size, pmaps5Size, pmaps6Size

	integer, pointer, dimension(:) :: pindSizes
	integer(4) :: pindSizesSize
		
	integer, pointer, dimension(:) :: pindOffs
	integer(4) :: pindOffsSize
		
	integer, pointer, dimension(:) :: pblkMap
	integer(4) :: pblkMapSize

	integer, pointer, dimension(:) :: poffset
	integer(4) :: poffsetSize

	integer, pointer, dimension(:) :: pnelems
	integer(4) :: pnelemsSize
		
	integer, pointer, dimension(:) :: pnthrcol
	integer(4) :: pnthrcolSize
		
	integer, pointer, dimension(:) :: pthrcol
	integer(4) :: pthrcolSize
	
	! variables for marshalling data from host to device memory and back
	integer(4) :: arg1Size, arg5Size, arg6Size
	real(8), dimension(:), pointer :: argument1
	real(8), dimension(:), pointer :: argument5
	real(8), dimension(:), pointer :: argument6
	
	logical :: isFirstTimeExecuting_adt_calc = .true.

	contains
	
		subroutine adt_calc_caller ( blockIdx, &
															 & argument1, pindMaps1, &
															 & pmaps1, &
															 & pmaps2, &
															 & pmaps3, &
															 & pmaps4, &
															 & argument5, &
															 & argument6, &
															 & pindSizes, &
															 & pindOffs, &
															 & blockOffset, &
															 & pblkMap, &
															 & poffset, &
															 & pnelems, &
															 & pnthrcol, &
															 & pthrcol &
														 & )

			integer(kind = OMP_integer_kind) :: blockIdx
			
			real(8), dimension(0:*) :: argument1
			
			integer(4), dimension(0:), target :: pindMaps1
			
			integer(2), dimension(0:*) :: pmaps1
			integer(2), dimension(0:*) :: pmaps2
			integer(2), dimension(0:*) :: pmaps3
			integer(2), dimension(0:*) :: pmaps4
		
		  real(8), dimension(0:*) :: argument5
			real(8), dimension(0:*) :: argument6
			
			integer(4), dimension(0:*) :: pindSizes
		
		
			integer(4), dimension(0:*) ::  pindOffs
			integer(4) :: blockOffset
			integer(4), dimension(0:*) :: pblkMap
			integer(4), dimension(0:*) :: poffset
			integer(4), dimension(0:*) :: pnelems
			integer(4), dimension(0:*) :: pnthrcol
			integer(4), dimension(0:*) :: pthrcol

			! local variables
			integer(4), pointer, dimension(:) :: pindArg1Map
			integer(4) :: pindArg1Size
			
			real(8), pointer, dimension(:) :: pindArg1Shared
			integer(4) :: nbytes0
			
			integer(4) :: nelem
			integer(4) :: offsetB

			integer(4) :: blockID
		
			integer(4) :: iter1
			integer(4) :: iter2
		
			real(8), dimension(0:8000-1), target :: sharedVirtual
		
!			print *, 'in the caller, blockIdx = ', blockIdx, ' and blockoffset = ', blockOffset
		
			! why this???
		  if ( 0 .eq. 0 ) then
			
				blockID = pblkMap ( blockIdx + blockOffset )

!				print *, 'Got blockID = ', blockID
				
				nelem = pnelems ( blockID )
				
!				print *, 'Got nelem'
				
				offsetB = poffset ( blockID )

!				print *, 'Got offset'

				
				pindArg1Size = pindSizes ( 0 + blockID * 1 )
				pindArg1Map => pindMaps1 ( 0 + pindOffs ( 0 + blockID * 1 ): )!(0 + pindOffs ( 0 + blockID * 1 )) + pindArg1Size  )

				! set shared memory pointers
				nbytes0 = 0

				pindArg1Shared => sharedVirtual ( nbytes0: )
			
			end if

			! copy indirect datasets into shared memory or zero increment
			do iter1 = 0, pindArg1Size-1
				do iter2 = 0, 2-1
				
					! + 1 because the arrays are in fortran notation
					pindArg1Shared ( (iter2 + iter1 * 2) + 1 ) = argument1 ( iter2 + pindArg1Map ( iter1 + 1 ) * 2 )
				
				end do
			end do
			
			do iter1 = 0, nelem-1
			
				call adt_calc ( pindArg1Shared ( 1 + pmaps1 ( iter1 + offsetB ) * 2: 1 + pmaps1 ( iter1 + offsetB ) * 2 + 2 - 1 ), &
											& pindArg1Shared ( 1 + pmaps2 ( iter1 + offsetB ) * 2: 1 + pmaps2 ( iter1 + offsetB ) * 2 + 2 - 1 ), &
											& pindArg1Shared ( 1 + pmaps3 ( iter1 + offsetB ) * 2: 1 + pmaps3 ( iter1 + offsetB ) * 2 + 2 - 1 ), &
											& pindArg1Shared ( 1 + pmaps4 ( iter1 + offsetB ) * 2: 1 + pmaps4 ( iter1 + offsetB ) * 2 + 2 - 1 ), &
											& argument5 ( (iter1 + offsetB) * 4: (iter1 + offsetB) * 4 + 4 - 1 ), &
											& argument6 ( (iter1 + offsetB) * 1: (iter1 + offsetB) * 1 + 1 - 1 ) &
										&	)
			end do

		end subroutine adt_calc_caller

	
		function op_par_loop_adt_calc ( subroutineName, setIn, &
																									 & arg1In,   idx1, map1In, access1, &
																									 & arg2In,   idx2, map2In, access2, &
																									 & arg3In,   idx3, map3In, access3, &
																									 & arg4In,   idx4, map4In, access4, &
																									 & arg5In,   idx5, map5In, access5, &
																									 & arg6In,   idx6, map6In, access6 &
																								 & )

			! use directives	
			use, intrinsic :: ISO_C_BINDING

			! mandatory	
			implicit none
			
			type(profInfo) :: op_par_loop_adt_calc
			
			
			! formal arguments
			character(kind=c_char,len=*), intent(in) :: subroutineName
			
			! data set on which we loop
			type(op_set), intent(in) :: setIn

			! data ids used in the function
			type(op_dat) :: arg1In, arg2In, arg3In, arg4In, arg5In, arg6In
			
			! index to be used in first and second pointers
			integer(4), intent(in) :: idx1, idx2, idx3, idx4, idx5, idx6
			
			! ptr ids for indirect access to data
			type(op_map) :: map1In, map2In, map3In, map4In, map5In, map6In
			
			! access values for arguments
			integer(4), intent(in) :: access1, access2, access3, access4, access5, access6
		
      ! local variables     
      type(op_set_core), pointer :: set
      type(op_map_core), pointer :: map1, map2, map3, map4, map5, map6
      type(op_dat_core), pointer :: arg1, arg2, arg3, arg4, arg5, arg6

      ! Compiler: variables used to invoke cplan 
			integer(4) :: args(6), idxs(6), maps(6), accs(6), inds(6), argsType(6)
			integer(4) :: argsNumber, indsNumber

			integer(4) :: iter

			! configuration variables for main kernel call
			integer(4) :: nblocks, nthread, nshared

			
			! number of threads
			integer(kind = OMP_integer_kind) :: nthreads = 0

			! thread index
			integer(kind = OMP_integer_kind) :: threadIndex = -1

			! block identifier in parallel loop
			integer(kind = OMP_integer_kind) :: blockIdx = -1

			! bounds of set slice assigned to each thread
			integer(kind = OMP_integer_kind) :: sliceStart = -1
			integer(kind = OMP_integer_kind) :: sliceEnd = -1

			! initialise timers
      real(kind=c_double) :: elapsedTimeStart = 0
      real(kind=c_double) :: elapsedTimeEnd = 0

      type(op_set_core), pointer :: arg1Set, arg5Set, arg6Set

      integer(4) :: partitionSize = 0

      ! initialising arguments
      set => setIn%setPtr
      
      map1 => map1In%mapPtr
      map2 => map2In%mapPtr
      map3 => map3In%mapPtr
      map4 => map4In%mapPtr
      map5 => map5In%mapPtr
      map6 => map6In%mapPtr
      
      arg1 => arg1In%dataPtr
      arg2 => arg2In%dataPtr
      arg3 => arg3In%dataPtr
      arg4 => arg4In%dataPtr
      arg5 => arg5In%dataPtr
      arg6 => arg6In%dataPtr

			if ( isFirstTimeExecuting_adt_calc .eqv. .true. ) then ! generate kernel input data

				isFirstTimeExecuting_adt_calc = .false.
			
				! get the plan
				args(1) = arg1%index
				args(2) = arg2%index
				args(3) = arg3%index
				args(4) = arg4%index
				args(5) = arg5%index
				args(6) = arg6%index
			
				idxs(1) = idx1
				idxs(2) = idx2
				idxs(3) = idx3
				idxs(4) = idx4
				idxs(5) = idx5
				idxs(6) = idx6
				
				! when passing from OP2 Fortran to OP2 C++ we have to decrement the idx values (not 1->N, but 0->N-1)
				! except -1 which indicates OP_ID or OP_GBL
				do iter = 1, 6
					if ( idxs(iter) /= -1 ) idxs(iter) = idxs(iter) - 1 
				end do

				maps(1) = map1%index
				maps(2) = map2%index
				maps(3) = map3%index
				maps(4) = map4%index
				maps(5) = map5%index
				maps(6) = map6%index

				accs(1) = access1
				accs(2) = access2
				accs(3) = access3
				accs(4) = access4
				accs(5) = access5
				accs(6) = access6


				! Compiler: generate this information by analysing the arguments
				argsNumber = 6
				indsNumber = 1 ! warning: this means the number of op_dat accessed indirectly, not the number of arguments!!
			
				inds(1) = 0
				inds(2) = 0
				inds(3) = 0
				inds(4) = 0
				inds(5) = -1
				inds(6) = -1

        if ( map1%dim .eq. -1 ) then ! global data
          argsType(1) = F_OP_ARG_GBL
        else
          argsType(1) = F_OP_ARG_DAT
        end if

        if ( map2%dim .eq. -1 ) then ! global data
          argsType(2) = F_OP_ARG_GBL
        else
          argsType(2) = F_OP_ARG_DAT
        end if

        if ( map3%dim .eq. -1 ) then ! global data
          argsType(3) = F_OP_ARG_GBL
        else
          argsType(3) = F_OP_ARG_DAT
        end if

        if ( map4%dim .eq. -1 ) then ! global data
          argsType(4) = F_OP_ARG_GBL
        else
          argsType(4) = F_OP_ARG_DAT
        end if

        if ( map5%dim .eq. -1 ) then ! global data
          argsType(5) = F_OP_ARG_GBL
        else
          argsType(5) = F_OP_ARG_DAT
        end if

        if ( map6%dim .eq. -1 ) then ! global data
          argsType(6) = F_OP_ARG_GBL
        else
          argsType(6) = F_OP_ARG_DAT
        end if

        partitionSize = 0


				! get the plan
				planRet = cplan_OpenMP ( subroutineName, &
                               & set%index, &
                               & argsNumber, &
                               & args, &
                               & idxs, &
                               & maps, &
                               & accs, &
                               & indsNumber, &
                               & inds, &
                               & argsType, &
                               & partitionSize &
                            &  )

				! now convert all pointers from C to Fortran
      
        call c_f_pointer( arg1%set, arg1Set )
				arg1Size = arg1%dim * arg1Set%size
				call c_f_pointer ( arg1%dat, argument1, (/arg1Size/) )
				
        call c_f_pointer( arg5%set, arg5Set )
				arg5Size = arg5%dim * arg5Set%size
				call c_f_pointer ( arg5%dat, argument5, (/arg5Size/) )
				
        call c_f_pointer( arg6%set, arg6Set )
				arg6Size = arg6%dim * arg6Set%size
				call c_f_pointer ( arg6%dat, argument6, (/arg6Size/) )
				
				! convert arguments and plan data to Fortran pointers (not a copy!)
				! Typically, first compute size, then transform pointer
				
				! transform the returned C pointer to a type(op_plan) variable
				call c_f_pointer ( planRet, actualPlan )		
				
				! convert nindirect  used to generate the pindMapsSize array of sizes
				call c_f_pointer ( actualPlan%nindirect, pnindirect, (/indsNumber/) )
					
				! convert pindMaps: there are indsNumber ind_maps
				call c_f_pointer ( actualPlan%ind_maps, pindMaps, (/indsNumber/) )
				
				! convert first position of the pindMaps array (the size is stored in the corresponding pnindirect position)
				call c_f_pointer ( pindMaps(1), pindMaps1, (/pnindirect(1)/) )
				! must be done for all indirect pointers: in this case just one op_dat is indirectly accessed)
				
				! convert maps in op_plan: there are argsNumber maps
				call c_f_pointer ( actualPlan%maps, pmaps, (/argsNumber/) )
				
				! convert positions in pmaps (only if the corresponding inds position is >= 0 (see op_support.cpp))
				! can't do a do-loop because I can't generate variable name
				if ( inds(1) .ge. 0 ) then
					pmaps1Size = set%size
					call c_f_pointer ( pmaps(1), pmaps1, (/pmaps1Size/) )
				end if
				
				if ( inds(2) .ge. 0 ) then
					pmaps2Size = set%size
					call c_f_pointer ( pmaps(2), pmaps2, (/pmaps2Size/) )
				end if
				
				if ( inds(3) .ge. 0 ) then
					pmaps3Size = set%size
					call c_f_pointer ( pmaps(3), pmaps3, (/pmaps3Size/) )
				end if
				
				if ( inds(4) .ge. 0 ) then
					pmaps4Size = set%size
					call c_f_pointer ( pmaps(4), pmaps4, (/pmaps4Size/) )
				end if
				
				if ( inds(5) .ge. 0 ) then
					pmaps5Size = set%size
					call c_f_pointer ( pmaps(5), pmaps5, (/pmaps5Size/) )
				end if
				
				if ( inds(6) .ge. 0 ) then
					pmaps6Size = set%size
					call c_f_pointer ( pmaps(6), pmaps6, (/pmaps6Size/) )
				end if	
				
				! converting ncolblk field to fortran variable
				call c_f_pointer ( actualPlan%ncolblk, ncolblk, (/set%size/) )
					
				! ind_sizes field has nblocks*indsNumber size
				pindSizesSize = actualPlan%nblocks * indsNumber
				call c_f_pointer ( actualPlan%ind_sizes, pindSizes, (/pindSizesSize/) )
					
				! ind_offset field has the same dimension of ind_sizes
				pindOffsSize = pindSizesSize
				call c_f_pointer ( actualPlan%ind_offs, pindOffs, (/pindOffsSize/) )
					
				! blkmap field has dimension nblocks
				pblkMapSize = actualPlan%nblocks
				call c_f_pointer ( actualPlan%blkmap, pblkMap, (/pblkMapSize/) )
					
				! offset field has dimension nblocks
				poffsetSize = actualPlan%nblocks
				call c_f_pointer ( actualPlan%offset, poffset, (/poffsetSize/) )
				
				! nelems field has dimension nblocks
				pnelemsSize = actualPlan%nblocks
				call c_f_pointer ( actualPlan%nelems, pnelems, (/pnelemsSize/) )

				! nthrcol field has dimension nblocks
				pnthrcolSize = actualPlan%nblocks
				call c_f_pointer ( actualPlan%nthrcol, pnthrcol, (/pnthrcolSize/) )
				
				! thrcol field has dimension set%size
				pthrcolSize = set%size
				call c_f_pointer ( actualPlan%thrcol, pthrcol, (/pthrcolSize/) )

			end if
			
 			! initialise timers
			call op_timers ( elapsedTimeStart )

      
!				print *, 'After converting stuff, nelems size = ', actualPlan%nblocks
			! get number of threads
#ifdef _OPENMP
			nthreads = omp_get_max_threads ()
#else
			nthreads = 1
#endif
			
			blockOffset = 0
			
			! execute the plan
			
!			print *, 'ncolblk(1) = ', ncolblk(1)
!			print *, 'ncolblk(2) = ', ncolblk(2)
!			print *, 'ncolblk(3) = ', ncolblk(3)
!			print *, 'NCOLORS = ', actualPlan%ncolors
			
			do col = 0, (actualPlan%ncolors - 1)
			
				! -1 is needed because the array variable is in Fortran notation
				nblocks = ncolblk(col+1)
				
				!$OMP PARALLEL DO
				do blockIdx = 0, nblocks-1
				
					call adt_calc_caller ( blockIdx, &
															 & argument1, pindMaps1, &
															 & pmaps1, &
															 & pmaps2, &
															 & pmaps3, &
															 & pmaps4, &
															 & argument5, &
															 & argument6, &
															 & pindSizes, &
															 & pindOffs, &
															 & blockOffset, &
															 & pblkMap, &
															 & poffset, &
															 & pnelems, &
															 & pnthrcol, &
															 & pthrcol &
														 & )
!					print *, 'blockoffset in the caller = ', blockOffset
				
				end do
			
				blockOffset = blockOffset + nblocks

			
			end do
			
!			print *, 'after the kernel call'
			
			! timing end
      call op_timers ( elapsedTimeEnd )

      op_par_loop_adt_calc%hostTime = elapsedTimeEnd - elapsedTimeStart

		end function op_par_loop_adt_calc


	


end module adt_calc_openmp