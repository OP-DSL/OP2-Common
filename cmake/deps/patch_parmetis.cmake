# patch_parmetis.cmake
# Patches METIS (bundled inside ParMETIS) to use 64-bit integer indices.
# Called with: cmake -DSRC=<SOURCE_DIR> -P patch_parmetis.cmake

file(READ "${SRC}/metis/include/metis.h" _content)
string(REPLACE "#define IDXTYPEWIDTH 32" "#define IDXTYPEWIDTH 64" _patched "${_content}")
if(_content STREQUAL _patched)
    message(STATUS "patch_parmetis: IDXTYPEWIDTH 32 not found - already patched or unexpected format")
else()
    file(WRITE "${SRC}/metis/include/metis.h" "${_patched}")
    message(STATUS "patch_parmetis: patched metis.h to IDXTYPEWIDTH 64")
endif()
