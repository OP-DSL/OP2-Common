# The example apps run to their own validation, every variant built, against
# downloaded meshes (label apps).  OP2_TEST_MGCFD_DIR names an MG-CFD-app-OP2
# checkout whose C++ app is built and run here too.

set(OP2_TEST_MESH_DIR "${CMAKE_BINARY_DIR}/test-meshes" CACHE PATH
    "Where the app tests' meshes are downloaded to, and kept between runs")
set(OP2_TEST_MGCFD_DIR "" CACHE PATH
    "An MG-CFD-app-OP2 checkout to build and test (empty: none)")

set(_op2_app_mpi_ranks_lowcost 8)
set(_op2_app_mpi_ranks_gpu 4)
set(_op2_app_omp_num_threads 6)

# Each mesh a test may read, and where it comes from.
set(_op2_ci_meshes "https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main")
set(_op2_mgcfd_meshes "https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software")
set(_op2_mesh_url_new_grid.dat "https://op-dsl.github.io/docs/OP2/new_grid.dat")
set(_op2_mesh_url_new_grid_dp.h5 "${_op2_ci_meshes}/airfoil/new_grid_dp.h5")
set(_op2_mesh_url_new_grid_sp.h5 "${_op2_ci_meshes}/airfoil/new_grid_sp.h5")
set(_op2_mesh_url_FE_grid.dat "${_op2_ci_meshes}/aero/FE_grid.dat")
set(_op2_mesh_url_FE_grid.h5 "${_op2_ci_meshes}/aero/FE_grid.h5")
set(_op2_mesh_url_m6_wing.tar.gz "${_op2_mgcfd_meshes}/m6_wing.tar.gz")
set(_op2_mesh_url_rotor37_1m.tar.gz "${_op2_mgcfd_meshes}/rotor37_1m.tar.gz")

set(_op2_meshes_used "")

# _op2_app_tests(<name> LANGUAGE cpp|fortran PASS <regex>
#                [SERIAL <target prefix>] [MPI <target prefix>]
#                [MESHES <mesh>[@<name read>]...] [FETCH <mesh archive>...]
#                [ARGS <arg>...])
#
# A test <name>_<variant> per serial target <SERIAL>_<variant> and
# <name>_mpi_<variant> per MPI target <MPI>_mpi_<variant> that exists.
# MESHES are linked into the run directory, each under the name the app
# opens; FETCH meshes are only downloaded, for an app given their directory.
function(_op2_app_tests name)
    cmake_parse_arguments(_A "" "LANGUAGE;PASS;SERIAL;MPI" "MESHES;FETCH;ARGS" ${ARGN})

    set(_links "")
    set(_meshes ${_A_FETCH})
    foreach(_mesh IN LISTS _A_MESHES)
        string(REPLACE "@" ";" _pair "${_mesh}")
        list(GET _pair 0 _file)
        list(GET _pair -1 _read)
        list(APPEND _links "${OP2_TEST_MESH_DIR}/${_file}@${_read}")
        list(APPEND _meshes "${_file}")
    endforeach()
    list(APPEND _op2_meshes_used ${_meshes})
    set(_op2_meshes_used "${_op2_meshes_used}" PARENT_SCOPE)
    list(TRANSFORM _meshes PREPEND op2_app_mesh_ OUTPUT_VARIABLE _fixtures)
    list(JOIN _links "|" _links)

    foreach(_variant IN LISTS _op2_variants_${_A_LANGUAGE})
        foreach(_form serial mpi)
            if(_form STREQUAL "serial" AND _A_SERIAL)
                set(_target "${_A_SERIAL}_${_variant}")
                set(_test "${name}_${_variant}")
                set(_cmd "$<TARGET_FILE:${_target}>")
                set(_nproc 1)
            elseif(_form STREQUAL "mpi" AND _A_MPI AND MPIEXEC_EXECUTABLE)
                set(_target "${_A_MPI}_mpi_${_variant}")
                set(_test "${name}_mpi_${_variant}")
                if(_variant MATCHES "^(cuda|hip)(_legacy|_f2c)?$")
                    set(_nproc ${_op2_app_mpi_ranks_gpu})
                else()
                    set(_nproc ${_op2_app_mpi_ranks_lowcost})
                endif()
                set(_cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_nproc}"
                    ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${_target}>" ${MPIEXEC_POSTFLAGS})
            else()
                continue()
            endif()
            if(NOT TARGET ${_target})
                continue()
            endif()

            list(APPEND _cmd ${_A_ARGS})
            list(JOIN _cmd "|" _cmd)
            add_test(NAME ${_test}
                COMMAND ${CMAKE_COMMAND}
                    "-DRUN_DIR=${CMAKE_CURRENT_BINARY_DIR}/${_test}"
                    "-DLINKS=${_links}"
                    "-DPASS=${_A_PASS}"
                    "-DCOMMAND=${_cmd}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/run_app.cmake")
            set(_env "")
            if(_variant MATCHES "^openmp(_f2c|_legacy)?$")
                set(_env "OMP_NUM_THREADS=${_op2_app_omp_num_threads}")
            endif()
            set_tests_properties(${_test} PROPERTIES
                LABELS "apps;${_A_LANGUAGE};${_form}"
                PROCESSORS ${_nproc}
                ENVIRONMENT "${_env}"
                FIXTURES_REQUIRED "${_fixtures}"
                TIMEOUT 1800)
        endforeach()
    endforeach()
endfunction()

set(_cpp_passed "This test is considered PASSED")

_op2_app_tests(airfoil_plain_dp LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL airfoil MPI airfoil_par MESHES new_grid.dat)
_op2_app_tests(airfoil_plain_sp LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL airfoil_sp MPI airfoil_sp_par MESHES new_grid.dat)
_op2_app_tests(airfoil_tempdats LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL airfoil_tempdats MPI airfoil_tempdats_par MESHES new_grid.dat)
_op2_app_tests(airfoil_hdf5_dp LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL airfoil_hdf5_dp MPI airfoil_hdf5_dp MESHES new_grid_dp.h5@new_grid.h5)
_op2_app_tests(airfoil_hdf5_sp LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL airfoil_hdf5_sp MPI airfoil_hdf5_sp MESHES new_grid_sp.h5@new_grid.h5)
_op2_app_tests(aero_plain LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL aero MPI aero_par MESHES FE_grid.dat)
_op2_app_tests(aero_hdf5 LANGUAGE cpp PASS "${_cpp_passed}"
    SERIAL aero_hdf5 MPI aero_hdf5 MESHES FE_grid.h5)

if(CMAKE_Fortran_COMPILER)
    _op2_app_tests(airfoil_plain_fortran LANGUAGE fortran PASS "Test PASSED"
        SERIAL airfoil_plain MESHES new_grid.dat)
    _op2_app_tests(airfoil_hdf5_fortran LANGUAGE fortran PASS "Test PASSED"
        SERIAL airfoil_hdf5 MPI airfoil_hdf5 MESHES new_grid_dp.h5@new_grid.h5)
endif()

# MG-CFD's C++ app, built from the checkout as its own CMake build would, run
# on two meshes it reads from their directories.
if(OP2_TEST_MGCFD_DIR)
    op2_add_app_variants(
        NAME     euler3d
        LANGUAGE cpp
        SOURCES  "${OP2_TEST_MGCFD_DIR}/euler3d.cpp"
        INCLUDE_DIRECTORIES "${OP2_TEST_MGCFD_DIR}"
        WITH_HDF5)

    foreach(_mesh m6_wing:M6_wing rotor37_1m:Rotor37_1M)
        string(REPLACE ":" ";" _mesh "${_mesh}")
        list(GET _mesh 0 _archive)
        list(GET _mesh 1 _dir)
        _op2_app_tests(mgcfd_${_archive} LANGUAGE cpp PASS "Validation passed"
            SERIAL euler3d MPI euler3d FETCH ${_archive}.tar.gz
            ARGS -i input.dat -d "${OP2_TEST_MESH_DIR}/${_dir}" -v)
    endforeach()
endif()

# A download per mesh, before the tests that read it run.
list(REMOVE_DUPLICATES _op2_meshes_used)
foreach(_mesh IN LISTS _op2_meshes_used)
    add_test(NAME op2_app_mesh_${_mesh}
        COMMAND ${CMAKE_COMMAND} "-DMESH_DIR=${OP2_TEST_MESH_DIR}"
                "-DMESHES=${_mesh}@${_op2_mesh_url_${_mesh}}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/fetch_meshes.cmake")
    set_tests_properties(op2_app_mesh_${_mesh} PROPERTIES
        LABELS apps
        FIXTURES_SETUP op2_app_mesh_${_mesh})
endforeach()
