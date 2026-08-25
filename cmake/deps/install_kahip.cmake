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

foreach(_lib IN LISTS _libs)
    # Fatal, not a warning: a partial install degrades quietly rather than
    # loudly. find_package(KaHIP) would just report KaHIP as not found and OP2
    # would configure without KaHIP partitioning, which is easy to miss in a
    # status line. The deps build asks for these targets by name, so a missing
    # one means the KaHIP build changed shape or failed.
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "install_kahip: ${_lib} was not built")
    endif()

    get_filename_component(_name "${_lib}" NAME)
    # Strip accidental double "lib" prefix from modified_kahip target name
    string(REPLACE "liblibmodified_kahip_interface.a"
                   "libmodified_kahip_interface.a" _name "${_name}")
    file(COPY_FILE "${_lib}" "${DST}/lib/${_name}")
endforeach()

file(COPY_FILE
    "${SOURCE_DIR}/interface/kaHIP_interface.h"
    "${DST}/include/kaHIP_interface.h")
file(COPY_FILE
    "${SOURCE_DIR}/parallel/parallel_src/interface/parhip_interface.h"
    "${DST}/include/parhip_interface.h")

message(STATUS "KaHIP installed to ${DST}")
