# The implicit proxy (apps/c/implicit) with each library built: from the mode,
# against the discrete solution; from the bump, by its true residual; and with
# four components under the Neumann preconditioner, which reads the matrix in
# solver memory.  MPI variants on two ranks, and there also from the mode on a
# randomly partitioned mesh, whose rows have as many blocks in other ranks'
# columns as in their own.

set(_libs "")
foreach(_def IN LISTS _op2_solve_defs)
    if(_def MATCHES "^OP2_SOLVE_(PETSC|HYPRE|GINKGO)$")
        string(TOLOWER "${CMAKE_MATCH_1}" _lib)
        list(APPEND _libs ${_lib})
    endif()
endforeach()

foreach(_nv IN LISTS OP2_IMPLICIT_NV)
    set(_cases "mode:-init:mode:-dt:0.002" "bump:-init:bump")
    if(_nv GREATER 1)
        list(APPEND _cases "neumann:-init:bump:-pc:neumann")
    endif()
    foreach(_variant seq genseq openmp cuda hip mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip)
        set(_target implicit${_nv}_${_variant})
        if(NOT TARGET ${_target})
            continue()
        endif()
        set(_cmd "$<TARGET_FILE:${_target}>")
        set(_variant_cases ${_cases})
        if(_variant MATCHES "^mpi_")
            set(_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${_target}>")
            list(APPEND _variant_cases "random:-init:mode:-dt:0.002:-partition:random")
        endif()
        foreach(_lib IN LISTS _libs)
            foreach(_case IN LISTS _variant_cases)
                string(REPLACE ":" ";" _args "${_case}")
                list(POP_FRONT _args _name)
                set(_solver "method=cg pc=jacobi rtol=1e-12 maxit=5000")
                if(_name STREQUAL "neumann")
                    set(_solver "method=cg rtol=1e-12 maxit=5000")
                endif()
                set(_test ${_target}_${_lib}_${_name})
                add_test(NAME ${_test} COMMAND ${_cmd} -n 24 -steps 3 ${_args} -solver "${_solver}" -check 1e-9)
                set_tests_properties(${_test} PROPERTIES
                    LABELS "solve" ENVIRONMENT "OP_SOLVE_LIB=${_lib};OMP_NUM_THREADS=2" TIMEOUT 300)
            endforeach()
        endforeach()
    endforeach()
endforeach()
