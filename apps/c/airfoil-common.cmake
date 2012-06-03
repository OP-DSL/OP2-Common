# This should only be loaded once
if(DEFINED __AIRFOIL_COMMON_INCLUDED)
  return()
endif()
set(__AIRFOIL_COMMON_INCLUDED TRUE)

include(GenerateMesh)
generate_mesh(AIRFOIL new_grid.dat naca0012)
