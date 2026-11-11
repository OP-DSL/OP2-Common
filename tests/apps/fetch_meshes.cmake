# Download each mesh an app test needs into MESH_DIR, skipping those already
# there; a .tar.gz is unpacked in place.  Run as the app tests' fixture:
#   cmake -DMESH_DIR=<dir> -DMESHES=<file>@<url>|... -P fetch_meshes.cmake

file(MAKE_DIRECTORY "${MESH_DIR}")
string(REPLACE "|" ";" meshes "${MESHES}")

foreach(mesh IN LISTS meshes)
    string(FIND "${mesh}" "@" at)
    string(SUBSTRING "${mesh}" 0 ${at} file)
    math(EXPR url_start "${at} + 1")
    string(SUBSTRING "${mesh}" ${url_start} -1 url)

    string(REGEX REPLACE "\\.tar\\.gz$" "" unpacked "${file}")
    if(EXISTS "${MESH_DIR}/${unpacked}")
        continue()
    endif()

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD "${url}" "${MESH_DIR}/${file}.part" STATUS status TLS_VERIFY ON)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        file(REMOVE "${MESH_DIR}/${file}.part")
        message(FATAL_ERROR "cannot download ${url}: ${status}")
    endif()
    file(RENAME "${MESH_DIR}/${file}.part" "${MESH_DIR}/${file}")

    if(NOT unpacked STREQUAL file)
        file(ARCHIVE_EXTRACT INPUT "${MESH_DIR}/${file}" DESTINATION "${MESH_DIR}")
        file(REMOVE "${MESH_DIR}/${file}")
    endif()
endforeach()
