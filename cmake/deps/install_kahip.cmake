# install_kahip.cmake
# Copies KaHIP static libraries and public headers from the build tree.
# KaHIP 3.x has no CMake install rules for static libs, so we do it manually.
#
# Called with:
#   cmake -DSRC=<BINARY_DIR> -DSOURCE_DIR=<SOURCE_DIR> -DDST=<kahip_prefix> -P install_kahip.cmake

file(MAKE_DIRECTORY "${DST}/lib" "${DST}/include")

set(_libs
    "${SRC}/libkahip_static.a"
    "${SRC}/parallel/modified_kahip/liblibmodified_kahip_interface.a"
    "${SRC}/parallel/parallel_src/libparhip_interface_static.a"
)

foreach(_lib ${_libs})
    if(EXISTS "${_lib}")
        get_filename_component(_name "${_lib}" NAME)
        # Strip accidental double "lib" prefix from modified_kahip target name
        string(REPLACE "liblibmodified_kahip_interface.a"
                       "libmodified_kahip_interface.a" _name "${_name}")
        file(COPY_FILE "${_lib}" "${DST}/lib/${_name}")
    else()
        message(WARNING "install_kahip: ${_lib} not found - skipping")
    endif()
endforeach()

file(COPY_FILE
    "${SOURCE_DIR}/interface/kaHIP_interface.h"
    "${DST}/include/kaHIP_interface.h")
file(COPY_FILE
    "${SOURCE_DIR}/parallel/parallel_src/interface/parhip_interface.h"
    "${DST}/include/parhip_interface.h")

message(STATUS "KaHIP installed to ${DST}")
