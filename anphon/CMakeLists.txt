cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O3") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0 -D_DEBUG")

project(anphon)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for ANPHON code (ALAMODE version: ${CMAKE_MATCH_1})")

if (FFTW3_ROOT)
    include_directories("${FFTW3_ROOT}/include")
    set(fftw3library "-L${FFTW3_ROOT}/lib -lfftw3")
elseif(NOT USE_MKL_FFT)
    set(fftw3library "-lfftw3")
else()
    set(mkl_root "$ENV{MKLROOT}")
    include_directories("${mkl_root}/include/fftw")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (openmplibrary OpenMP::OpenMP_CXX)
endif()

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    if (BOOST_INCLUDE)
        include_directories(${BOOST_INCLUDE})
    else()
        message(FATAL_ERROR "Boost was not found.
        Please specify the location of boost include directories
        via -DBOOST_INCLUDE option.")
    endif()
endif()

find_package(Eigen3)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    if (EIGEN3_INCLUDE)
        include_directories(${EIGEN3_INCLUDE})
    else()
        message(FATAL_ERROR "Eigen3 was not found.
        Please specify the location of boost include directories
        via -DEIGEN3_INCLUDE option.")
    endif()
endif()

find_package(LAPACK REQUIRED)
include_directories(${Lapack_INCLUDE_DIRS})
find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_DIRS})

if (WITH_HDF5_SUPPORT)
    add_definitions(-D_HDF5)
    find_package(HDF5 COMPONENTS CXX)
    if (HDF5_FOUND)
        include_directories(${HDF5_INCLUDE_DIRS})
        set(hdf5library ${HDF5_LIBRARIES})
    else()
        if (HDF5_ROOT)
            include_directories("${HDF5_ROOT}/include")
            set(hdf5library "-L${HDF5_ROOT}/lib -lhdf5_cpp -lhdf5")
        else()
            message(FATAL_ERROR "HDF5 was not found.
            Please specify the HDF5 install directory
            via -DHDF5_ROOT option.")
        endif()
    endif()
endif()

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

# Source code
include_directories("${PROJECT_SOURCE_DIR}/../include")
set(SOURCES ${PROJECT_SOURCE_DIR}/anharmonic_core.cpp
            ${PROJECT_SOURCE_DIR}/conductivity.cpp
            ${PROJECT_SOURCE_DIR}/dielec.cpp
            ${PROJECT_SOURCE_DIR}/dynamical.cpp
            ${PROJECT_SOURCE_DIR}/ewald.cpp
            ${PROJECT_SOURCE_DIR}/fcs_phonon.cpp
            ${PROJECT_SOURCE_DIR}/gruneisen.cpp
            ${PROJECT_SOURCE_DIR}/integration.cpp
            ${PROJECT_SOURCE_DIR}/isotope.cpp
            ${PROJECT_SOURCE_DIR}/kpoint.cpp
            ${PROJECT_SOURCE_DIR}/main.cpp
            ${PROJECT_SOURCE_DIR}/mode_analysis.cpp
            ${PROJECT_SOURCE_DIR}/mpi_common.cpp
            ${PROJECT_SOURCE_DIR}/parsephon.cpp
            ${PROJECT_SOURCE_DIR}/phonon_dos.cpp
            ${PROJECT_SOURCE_DIR}/phonon_velocity.cpp
            ${PROJECT_SOURCE_DIR}/phonons.cpp
            ${PROJECT_SOURCE_DIR}/scph.cpp
            ${PROJECT_SOURCE_DIR}/selfenergy.cpp
            ${PROJECT_SOURCE_DIR}/symmetry_core.cpp
            ${PROJECT_SOURCE_DIR}/system.cpp
            ${PROJECT_SOURCE_DIR}/thermodynamics.cpp
            ${PROJECT_SOURCE_DIR}/timer.cpp
            ${PROJECT_SOURCE_DIR}/write_phonons.cpp)

set(HEADERS ${PROJECT_SOURCE_DIR}/anharmonic_core.h
            ${PROJECT_SOURCE_DIR}/conductivity.h
            ${PROJECT_SOURCE_DIR}/dielec.h
            ${PROJECT_SOURCE_DIR}/dynamical.h
            ${PROJECT_SOURCE_DIR}/error.h
            ${PROJECT_SOURCE_DIR}/ewald.h
            ${PROJECT_SOURCE_DIR}/fcs_phonon.h
            ${PROJECT_SOURCE_DIR}/gruneisen.h
            ${PROJECT_SOURCE_DIR}/integration.h
            ${PROJECT_SOURCE_DIR}/isotope.h
            ${PROJECT_SOURCE_DIR}/kpoint.h
            ${PROJECT_SOURCE_DIR}/memory.h
            ${PROJECT_SOURCE_DIR}/mode_analysis.h
            ${PROJECT_SOURCE_DIR}/mpi_common.h
            ${PROJECT_SOURCE_DIR}/parsephon.h
            ${PROJECT_SOURCE_DIR}/phonon_dos.h
            ${PROJECT_SOURCE_DIR}/phonon_velocity.h
            ${PROJECT_SOURCE_DIR}/phonons.h
            ${PROJECT_SOURCE_DIR}/pointers.h
            ${PROJECT_SOURCE_DIR}/scph.h
            ${PROJECT_SOURCE_DIR}/selfenergy.h
            ${PROJECT_SOURCE_DIR}/symmetry_core.h
            ${PROJECT_SOURCE_DIR}/system.h
            ${PROJECT_SOURCE_DIR}/thermodynamics.h
            ${PROJECT_SOURCE_DIR}/timer.h
            ${PROJECT_SOURCE_DIR}/write_phonons.h)

message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
message("MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
# Executable
add_executable(anphon ${SOURCES} ${HEADERS})
target_link_libraries(anphon ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${openmplibrary})
if (WITH_HDF5_SUPPORT)
    target_link_libraries(anphon ${hdf5library})
endif()
if (fftw3library)
    target_link_libraries(anphon ${fftw3library})
endif()
