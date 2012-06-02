message(STATUS "Setting Intel Fortran compiler options")

SET(CMAKE_Fortran_FLAGS_DIAGS_INIT "-warn all -warn error")

include(Compiler/Intel)
__compiler_intel(Fortran)

SET(CMAKE_Fortran_MODDIR_FLAG "-module ")
