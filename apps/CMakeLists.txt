# Dispatcher for OP2 example apps.  Each subdirectory that has a CMakeLists.txt
# defines its own app variants.  Unported apps still use the legacy GNU Make
# files in makefiles/{c,f}_app.mk.

add_subdirectory(c/aero)
add_subdirectory(c/airfoil)
add_subdirectory(c/jac1)
add_subdirectory(c/jac2)
add_subdirectory(c/min)
add_subdirectory(c/reduction)

if(CMAKE_Fortran_COMPILER)
    add_subdirectory(fortran/airfoil)
    add_subdirectory(fortran/jac1)
    add_subdirectory(fortran/jac1_long)
    add_subdirectory(fortran/reduction)
endif()
