cmake_minimum_required(VERSION 3.16)
project(subdwarf_rv_simulation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- default to Release build with aggressive optimizations ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# optional: LTO (link-time optimization) for even more inlining
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)
if(LTO_SUPPORTED)
    message(STATUS "LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# ---- dependencies ----
find_package(Boost REQUIRED COMPONENTS program_options iostreams)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

find_path(GNUPLOT_IOSTREAM_INCLUDE_DIR
    NAMES gnuplot-iostream.h
    HINTS
        ${CMAKE_INCLUDE_PATH}
        $ENV{HOME}/Projects
        /usr/include
        /usr/local/include
    DOC "Directory containing gnuplot-iostream.h"
)

if(NOT GNUPLOT_IOSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find gnuplot-iostream.h.\n"
        "Set -DGNUPLOT_IOSTREAM_DIR=/path/to/dir or add it to CMAKE_INCLUDE_PATH.")
else()
    message(STATUS "Found gnuplot-iostream.h in: ${GNUPLOT_IOSTREAM_INCLUDE_DIR}")
endif()

# ---- library of reusable source files ----
add_library(rv_lib STATIC
    src/models.cpp
    src/maths.cpp
    src/vector_operations.cpp
    src/file_io.cpp
    src/lomb_scargle_periodogram.cpp
    src/exceptions.cpp
)

target_include_directories(rv_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GNUPLOT_IOSTREAM_INCLUDE_DIR}
)

target_link_libraries(rv_lib PUBLIC
    Boost::program_options
    Boost::iostreams
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    PkgConfig::FFTW3
)

# ---- main executable ----
add_executable(rv_mcmc src/rv_mcmc.cpp)
target_link_libraries(rv_mcmc PRIVATE rv_lib)