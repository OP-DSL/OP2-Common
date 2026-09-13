cmake_minimum_required(VERSION 3.26)
project(op2-flang-scan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# find_package(Flang) is not called here, as the packaged FlangConfig.cmake
# unconditionally imports clangBasic / clangDriver / clangOptions, so it fails
# when LLVM was built without "clang" in LLVM_ENABLE_PROJECTS  

# LLVMExports.cmake may require zstd::libzstd_shared
# Distro LLVM packages often ship libzstd1 without zstd, which breaks generation
if(NOT TARGET zstd::libzstd_shared)
    find_library(OP2_ZSTD_LIBRARY NAMES zstd)
    if(NOT OP2_ZSTD_LIBRARY)
        foreach(_zstd_cand
                /usr/lib/x86_64-linux-gnu/libzstd.so
                /usr/lib/x86_64-linux-gnu/libzstd.so.1
                /usr/lib64/libzstd.so
                /usr/lib64/libzstd.so.1
                /usr/lib/libzstd.so
                /usr/lib/libzstd.so.1)
            if(EXISTS "${_zstd_cand}")
                set(OP2_ZSTD_LIBRARY "${_zstd_cand}")
                break()
            endif()
        endforeach()
    endif()
    if(OP2_ZSTD_LIBRARY)
        add_library(zstd::libzstd_shared SHARED IMPORTED)
        set_target_properties(zstd::libzstd_shared PROPERTIES
            IMPORTED_LOCATION "${OP2_ZSTD_LIBRARY}")
        message(STATUS "Providing zstd::libzstd_shared at ${OP2_ZSTD_LIBRARY}")
    endif()
endif()

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

get_filename_component(LLVM_INSTALL_PREFIX "${LLVM_INCLUDE_DIRS}" DIRECTORY)
set(FLANG_INCLUDE_DIRS "${LLVM_INSTALL_PREFIX}/include")
set(FLANG_LIBRARY_DIR  "${LLVM_INSTALL_PREFIX}/lib")

# Sanity check that Flang headers are installed beside LLVM
if(NOT EXISTS "${FLANG_INCLUDE_DIRS}/flang/Parser/parsing.h")
    message(FATAL_ERROR "Flang parser headers not found at ${FLANG_INCLUDE_DIRS}/flang/Parser")
endif()

include_directories(SYSTEM
    ${LLVM_INCLUDE_DIRS}
    ${FLANG_INCLUDE_DIRS}
)

link_directories(${FLANG_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

# Flang parser headers require an endianness macro
if(CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    add_compile_definitions(FLANG_BIG_ENDIAN=1)
else()
    add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
endif()

add_executable(op2-flang-scan op2-flang-scan.cpp)

set(OP2_FLANG_LIBS FortranParser)
foreach(lib IN ITEMS FortranCommon FortranSupport FortranEvaluate FortranDecimal)
    if(EXISTS "${FLANG_LIBRARY_DIR}/lib${lib}.a"
       OR EXISTS "${FLANG_LIBRARY_DIR}/lib${lib}.so")
        list(APPEND OP2_FLANG_LIBS ${lib})
    endif()
endforeach()
message(STATUS "Linking Flang libraries: ${OP2_FLANG_LIBS}")

target_link_libraries(op2-flang-scan PRIVATE ${OP2_FLANG_LIBS})
target_link_libraries(op2-flang-scan PRIVATE LLVMSupport)

set(OP2_LLVM_FRONTEND_LIBS)
foreach(lib IN ITEMS LLVMFrontendOpenMP LLVMFrontendOpenACC LLVMFrontendDirective LLVMFrontendHLSL LLVMTargetParser)
    if(EXISTS "${FLANG_LIBRARY_DIR}/lib${lib}.a"
       OR EXISTS "${FLANG_LIBRARY_DIR}/lib${lib}.so")
        list(APPEND OP2_LLVM_FRONTEND_LIBS ${lib})
    endif()
endforeach()
message(STATUS "Linking LLVM frontend helper libraries: ${OP2_LLVM_FRONTEND_LIBS}")
target_link_libraries(op2-flang-scan PRIVATE ${OP2_LLVM_FRONTEND_LIBS})

install(TARGETS op2-flang-scan RUNTIME DESTINATION bin)
