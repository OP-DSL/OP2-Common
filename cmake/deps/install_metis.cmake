# install_metis.cmake
# Copies the METIS library and header from the ParMETIS cmake build tree.
# ParMETIS 4.0.3 has no CMake install rules for METIS, so we do it manually.
#
# Called with:
#   cmake -DSRC=<BINARY_DIR> -DSOURCE_DIR=<SOURCE_DIR> -DDST=<metis_prefix> -P install_metis.cmake

file(MAKE_DIRECTORY "${DST}/lib" "${DST}/include")

# libmetis.a lives inside the METIS subdirectory of the ParMETIS cmake build
file(GLOB_RECURSE _candidates
    "${SRC}/metis/libmetis/libmetis.a"
    "${SRC}/libmetis/libmetis.a"
    "${SRC}/libmetis.a"
)
if(NOT _candidates)
    file(GLOB_RECURSE _candidates "${SRC}/**/libmetis.a")
endif()
if(NOT _candidates)
    message(FATAL_ERROR "install_metis: could not find libmetis.a under ${SRC}")
endif()

list(GET _candidates 0 _libmetis)
file(COPY_FILE "${_libmetis}" "${DST}/lib/libmetis.a")
file(COPY_FILE "${SOURCE_DIR}/metis/include/metis.h" "${DST}/include/metis.h")
message(STATUS "METIS installed to ${DST}")
