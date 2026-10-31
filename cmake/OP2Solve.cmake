# The solver layer (op2/src/solve) and the libraries it can solve with. Any
# one enabled compiles the layer into every C library variant, linked against
# the library builds found here; a GPU-enabled PETSc, hypre or Ginkgo also
# serves the host variants.

option(OP2_ENABLE_PETSC  "Solver layer: PETSc backend (PETSC_DIR, PETSC_ARCH)" OFF)
option(OP2_ENABLE_HYPRE  "Solver layer: hypre backend (HYPRE_ROOT)"            OFF)
option(OP2_ENABLE_GINKGO "Solver layer: Ginkgo backend (Ginkgo_ROOT)"          OFF)

set(_op2_solve_defs "")
set(_op2_solve_libs "")
set(_op2_solve_srcs "")

if(OP2_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    set(PETSC_DIR  "$ENV{PETSC_DIR}"  CACHE PATH   "PETSc directory")
    set(PETSC_ARCH "$ENV{PETSC_ARCH}" CACHE STRING "PETSc arch within PETSC_DIR")
    set(ENV{PKG_CONFIG_PATH} "${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
    message(STATUS "OP2: PETSc        = ${PETSC_VERSION} (${PETSC_DIR}/${PETSC_ARCH})")
    list(APPEND _op2_solve_defs OP2_SOLVE_PETSC)
    list(APPEND _op2_solve_libs PkgConfig::PETSC)
    list(APPEND _op2_solve_srcs op_solve_petsc.cpp)
endif()

if(OP2_ENABLE_HYPRE)
    find_package(HYPRE REQUIRED CONFIG)
    message(STATUS "OP2: hypre        = ${HYPRE_VERSION} (${HYPRE_DIR})")
    list(APPEND _op2_solve_defs OP2_SOLVE_HYPRE)
    list(APPEND _op2_solve_libs HYPRE::HYPRE)
    list(APPEND _op2_solve_srcs op_solve_hypre.cpp)
endif()

if(OP2_ENABLE_GINKGO)
    find_package(Ginkgo REQUIRED CONFIG)
    message(STATUS "OP2: Ginkgo       = ${Ginkgo_VERSION} (${Ginkgo_DIR})")
    list(APPEND _op2_solve_defs OP2_SOLVE_GINKGO)
    list(APPEND _op2_solve_libs Ginkgo::ginkgo)
    list(APPEND _op2_solve_srcs op_solve_ginkgo.cpp)
endif()

if(_op2_solve_srcs)
    list(APPEND _op2_solve_defs OP2_SOLVE)
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "OP2: the solver layer needs MPI (the solver libraries are MPI builds)")
    endif()
    list(PREPEND _op2_solve_srcs op_solve.cpp op_solve_layout.cpp)
endif()

# The layer and its libraries into one C library target.
function(_op2_link_solve tgt)
    if(_op2_solve_srcs)
        target_compile_definitions(${tgt} PRIVATE ${_op2_solve_defs})
        target_link_libraries(${tgt} PRIVATE ${_op2_solve_libs} MPI::MPI_CXX)
    endif()
endfunction()
