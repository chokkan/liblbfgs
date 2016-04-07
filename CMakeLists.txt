# ==============================================================================
# libLBFGS: C library of limited-memory BFGS (L-BFGS)
#
# Copyright (c) 1990, Jorge Nocedal
# Copyright (c) 2007-2010, Naoaki Okazaki
#
# libLBFGS is distributed under the term of the MIT license.
# Please refer to COPYING file in the distribution.
# ==============================================================================

#.rst
# CMake configuration of libLBFGS project
# ---------------------------------------
#
# This CMakeLists.txt defines some libLBFGS specific configuration variables
# using the "liblbfgs_define" utility macro. The default values of these variables
# can be overridden either on the CMake command-line using the -D option of
# the cmake command or in a super-project which includes the libLBFGS source
# tree by setting the LibLBFGS_<varname> CMake variables before adding the
# libLBFGS source directory via CMake's "add_subdirectory" command. Only when
# the non-cached variable LibLBFGS_IS_SUBPROJECT has a value equivalent to FALSE,
# these configuration variables are added to the CMake cache so they can be
# edited in the CMake GUI. By default, LibLBFGS_IS_SUBPROJECT is set to TRUE when
# the CMAKE_SOURCE_DIR is not identical to the directory of this CMakeLists.txt
# file, i.e., the top-level directory of the libLBFGS project source tree.
#
# Example CMakeLists.txt of user project which requires separate libLBFGS
# installation (possibly requires FindLibLBFGS.cmake module for distribution
# packages of libLBFGS that do not include a LibLBFGSConfig.cmake file)::
#
#    cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
#
#    project(Foo)
#
#    find_package(LibLBFGS REQUIRED)
#
#    add_executable(foo src/foo.cc)
#    target_link_libraries(foo liblbfgs)
#
# Example CMakeLists.txt of super-project which contains libLBFGS source tree::
#
#    cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
#
#    project(Foo)
#
#    set(LibLBFGS_USE_SSE ON)
#    add_subdirectory(lbfgs)
#
#    add_executable(foo src/foo.cc)
#    target_link_libraries(foo liblbfgs)
#
# Variables to configure the source files::
#
#    LibLBFGS_USE_DOUBLE  - Enable double precision floating point arithmetics. (default: ON)
#    LibLBFGS_USE_SSE     - Enable SSE/SSE2 optimiations. (default: OFF)
#    LibLBFGS_USE_IEEE754 - Enable optimization routines for IEEE754 floating point values. (default: ON)
#
# Variables to configure the build::
#
#    LibLBFGS_BUILD_SHARED_LIBS - Enable build of shared libraries. (default: OFF)
#    LibLBFGS_BUILD_EXAMPLES    - Enable build of example programs. (default: OFF)
#
# Variables to configure the installation::
#
#    LibLBFGS_INSTALL_STATIC_LIBS - Whether to install static library files.
#                                   Shared libraries are always installed.
#                                   When a library is installed, its public header
#                                   files are installed as well. The default is
#                                   to not install static libraries when this
#                                   project is a subproject of another project.
#    LibLBFGS_INSTALL_CONFIG      - Whether to install CMake configuration files.
#                                   By default, the CMake configuration files are
#                                   installed when the library itself is installed.
#    LibLBFGS_INSTALL_RUNTIME_DIR - Installation directory for runtime files. (default: bin)
#    LibLBFGS_INSTALL_INCLUDE_DIR - Installation directory for public header files. (default: include)
#    LibLBFGS_INSTALL_LIBRARY_DIR - Installation directory for library files. (default: lib)
#    LibLBFGS_INSTALL_CONFIG_DIR  - Installation directory for CMake configuration. (default: lib/cmake/liblbfgs)

# ----------------------------------------------------------------------------
# CMake version and policies
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif ()

# ----------------------------------------------------------------------------
# includes
include(CheckIncludeFile)
include(CMakePackageConfigHelpers)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Subproject.cmake")

# ----------------------------------------------------------------------------
# package
subproject(LibLBFGS VERSION 1.10.0 LANGUAGES C)

set(PACKAGE_STRING    "${PROJECT_NAME} ${PROJECT_VERSION}")
set(PACKAGE_TARNAME   "${PROJECT_NAME_LOWER}-${PROJECT_VERSION}")
set(PACKAGE_BUGREPORT "https://github.com/chokkan/liblbfgs/issues")

# ----------------------------------------------------------------------------
# options
define(BOOL BUILD_SHARED_LIBS "Enable build of shared libraries"                OFF)
define(BOOL BUILD_EXAMPLES    "Enable build of sample programs"                 OFF)
define(BOOL USE_DOUBLE        "Use double precision floating point arithmetics" ON)
define(BOOL USE_SSE           "Use SSE/SSE2 instructions for optimization"      OFF)
define(BOOL USE_IEEE754       "Activate optimization routines for IEEE754 floating point values" ON)

set_cache_property(USE_IEEE754 ADVANCED TRUE)

# ----------------------------------------------------------------------------
# checks for SSE/SSE2 instructions header files
if (USE_SSE)
  if (USE_DOUBLE)
    check_include_file(emmintrin.h HAVE_EMMINTRIN_H)
    if (NOT HAVE_EMMINTRIN_H)
      message(WARNING "SSE2 instructions header file emmintrin.h not found. Disabled SSE optimizations.")
      set_value(USE_SSE OFF)
    endif ()
  else ()
    check_include_file(xmmintrin.h HAVE_XMMINTRIN_H)
    if (NOT HAVE_XMMINTRIN_H)
      message(WARNING "SSE instructions header file xmmintrin.h not found. Disabled SSE optimizations.")
      set_value(USE_SSE OFF)
    endif ()
  endif ()
endif ()

# ----------------------------------------------------------------------------
# library
set(HEADERS
  "include/lbfgs.h"
)

set(SOURCES
  "lib/lbfgs.c"
  "lib/arithmetic_ansi.h"
  "lib/arithmetic_sse_float.h"
  "lib/arithmetic_sse_double.h"
)

add_library(liblbfgs ${HEADERS} ${SOURCES})

set_target_properties(liblbfgs PROPERTIES
  OUTPUT_NAME lbfgs
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_SOVERSION}
  DEBUG_POSTFIX d
  POSITION_INDEPENDENT_CODE TRUE
)

target_include_directories(liblbfgs
  PUBLIC  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  PRIVATE "${PROJECT_SOURCE_DIR}/lib"
)

if (NOT USE_DOUBLE)
  target_compile_definitions(liblbfgs INTERFACE LBFGS_FLOAT=32)
endif ()
if (NOT USE_IEEE754)
  target_compile_definitions(liblbfgs INTERFACE LBFGS_IEEE_FLOAT=0)
endif ()
if (USE_SSE)
  if (USE_DOUBLE)
    target_compile_definitions(liblbfgs PRIVATE HAVE_EMMINTRIN_H=${HAVE_EMMINTRIN_H})
    if (MSVC)
      target_compile_definitions(liblbfgs PRIVATE __SSE2__)
    endif ()
  else ()
    target_compile_definitions(liblbfgs PRIVATE HAVE_XMMINTRIN_H=${HAVE_XMMINTRIN_H})
    if (MSVC)
      target_compile_definitions(liblbfgs PRIVATE __SSE__)
    endif ()
  endif ()
endif ()

if (CMAKE_COMPILER_IS_GNUC)
  target_compile_options(liblbfgs PRIVATE "$<$<CONFIG:Release>:-ffast-math>")
  if (USE_SSE)
    if (USE_DOUBLE)
      target_compile_options(liblbfgs PRIVATE "-msse2")
    else ()
      target_compile_options(liblbfgs PRIVATE "-msse")
    endif ()
  endif ()
endif ()

install_library(liblbfgs
  RUNTIME_DESTINATION "bin"
  LIBRARY_DESTINATION "lib"
  INCLUDE_DESTINATION "include"
  PUBLIC_HEADER_FILES "${HEADERS}"
)

# ----------------------------------------------------------------------------
# examples
if (BUILD_EXAMPLES)
  add_executable(sample sample/sample.c)
  target_link_libraries(sample liblbfgs)
endif ()

# ----------------------------------------------------------------------------
# configuration
if (${PROJECT_NAME}_INSTALL_CONFIG_DIR)
  set(PROJECT_INSTALL_CONFIG_DIR "${${PROJECT_NAME}_INSTALL_CONFIG_DIR}")
elseif (WIN32 AND NOT MINGW AND NOT CYGWIN)
  set(PROJECT_INSTALL_CONFIG_DIR "cmake")
else ()
  set(PROJECT_INSTALL_CONFIG_DIR "lib/cmake/${PROJECT_NAME_LOWER}")
endif ()

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${PROJECT_INSTALL_CONFIG_DIR}"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION       ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install_package_config_files(
  FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
    "${PROJECT_INSTALL_CONFIG_DIR}"
)

export(TARGETS liblbfgs FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
install_exports(DESTINATION "${PROJECT_INSTALL_CONFIG_DIR}")
