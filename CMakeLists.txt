cmake_minimum_required(VERSION 3.16)
project(rv_mcmc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RV_MCMC_BUILD_CLI    "Build standalone CLI (rv_mcmc.cpp)"  OFF)
option(RV_MCMC_BUILD_PYBIND "Build python bindings"               OFF)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(FFTW3  REQUIRED)        # or pkg_check_modules

# ----- core library (no main, no boost::program_options) -----
add_library(rv_mcmc STATIC
    src/exceptions.cpp
    src/file_io.cpp
    src/lomb_scargle_periodogram.cpp
    src/maths.cpp
    src/models.cpp
    src/vector_operations.cpp
    src/rv_mcmc_api.cpp
)
target_include_directories(rv_mcmc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rv_mcmc
    PUBLIC  Eigen3::Eigen
    PRIVATE OpenMP::OpenMP_CXX fftw3
)

# Optional: legacy CLI (only compiled if requested)
if (RV_MCMC_BUILD_CLI)
    find_package(Boost REQUIRED COMPONENTS program_options)
    add_executable(rv_mcmc_cli src/rv_mcmc.cpp)
    target_link_libraries(rv_mcmc_cli PRIVATE rv_mcmc Boost::program_options)
endif()

target_compile_definitions(rv_mcmc PRIVATE) 