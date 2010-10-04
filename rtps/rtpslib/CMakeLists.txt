IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(rtpslib C CXX)

#change this later
SET( BASE_DIR ${rtpslib_SOURCE_DIR}/../..)

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${BASE_DIR}/cmake
     )

message("${CMAKE_MODULE_PATH}\n")

#Only define this on the apple platform
# appears in RadixSort.cl
IF(APPLE)
	ADD_DEFINITIONS(-DMAC)
ENDIF()

#FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

set (rtpslib_INCLUDES
    ${rtpslib_SOURCE_DIR}/oclRadixSortKeysValues/inc
    ${rtpslib_SOURCE_DIR}/bitonic_sort/src
    ${rtpslib_SOURCE_DIR}/opencl
    ${rtpslib_SOURCE_DIR}/
    ${rtpslib_SOURCE_DIR}/system
    #${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${BASE_DIR}/opencl10
	$ENV{OPENCL_HOME}/common/inc
	$ENV{OPENCL_HOME}/../shared/inc
	${rtpslib_SOURCE_DIR}/oclRadixSortKeysalues/inc
    ${rtpslib_SOURCE_DIR}/ge_datastructures
)
message("rtpslib_SOURCE_DIR: ${rtpslib_SOURCE_DIR}")


#library
set (rtpslib_SOURCES
    ${rtpslib_SOURCE_DIR}/timege.cpp
    #${rtpslib_SOURCE_DIR}/ge_datastructures/datastructures.cpp
	${rtpslib_SOURCE_DIR}/oclRadixSortKeysValues/src/RadixSort.cpp
	${rtpslib_SOURCE_DIR}/oclRadixSortKeysValues/src/Scan.cpp
	${rtpslib_SOURCE_DIR}/bitonic_sort/src/oclBitonicSort_launcher.cpp

    # "wrap" makes it clear we are talking about c++ code
	# each file wraps a single device kernel and calls it
	${rtpslib_SOURCE_DIR}/system/utils/build_datastructures_wrap.cpp
	${rtpslib_SOURCE_DIR}/system/utils/neighbor_search_wrap.cpp
	${rtpslib_SOURCE_DIR}/system/utils/hash_wrap.cpp
	${rtpslib_SOURCE_DIR}/system/utils/sort_wrap.cpp
	${rtpslib_SOURCE_DIR}/system/utils/bitonic_sort_wrap.cpp
	${rtpslib_SOURCE_DIR}/system/utils/collision_wall_wrap.cpp

    ${rtpslib_SOURCE_DIR}/RTPS.cpp
    ${rtpslib_SOURCE_DIR}/RTPSettings.cpp
    ${rtpslib_SOURCE_DIR}/util.cpp

    ${rtpslib_SOURCE_DIR}/opencl/CLL.cpp
    ${rtpslib_SOURCE_DIR}/opencl/Kernel.cpp

    ${rtpslib_SOURCE_DIR}/render/Render.cpp
    ${rtpslib_SOURCE_DIR}/particle/UniformGrid.cpp
    ${rtpslib_SOURCE_DIR}/system/Simple.cpp
    ${rtpslib_SOURCE_DIR}/system/GE_SPH.cpp     # NEW

    ${rtpslib_SOURCE_DIR}/system/ge_sph/Euler.cpp
    ${rtpslib_SOURCE_DIR}/system/ge_sph/Density.cpp
    ${rtpslib_SOURCE_DIR}/system/ge_sph/Pressure.cpp
    ${rtpslib_SOURCE_DIR}/system/ge_sph/Viscosity.cpp




    #${rtpslib_SOURCE_DIR}/system/SPH.cpp
    #${rtpslib_SOURCE_DIR}/opencl/Buffer.cpp

    #${rtpslib_SOURCE_DIR}/system/sph/Euler.cpp
    #${rtpslib_SOURCE_DIR}/system/sph/Density.cpp
    #${rtpslib_SOURCE_DIR}/system/sph/Pressure.cpp
    #${rtpslib_SOURCE_DIR}/system/sph/Viscosity.cpp
    #${rtpslib_SOURCE_DIR}/system/sph/Collision_wall.cpp
    )


ADD_DEFINITIONS(-g )
ADD_DEFINITIONS(-DGPU)
ADD_DEFINITIONS(-DCL_SORT_SOURCE_DIR="${rtpslib_SOURCE_DIR}/oclRadixSortKeysValues/")
ADD_DEFINITIONS(-DCL_BITONIC_SORT_SOURCE_DIR="${rtpslib_SOURCE_DIR}/bitonic_sort/")
ADD_DEFINITIONS(-DCL_SOURCE_DIR="${rtpslib_SOURCE_DIR}/system/sph/")
#ADD_DEFINITIONS(-DCL_UTIL_SOURCE_DIR="${rtpslib_SOURCE_DIR}/gordon_erle/datastructures_cl")
ADD_DEFINITIONS(-DCL_SPH_SOURCE_DIR="${rtpslib_SOURCE_DIR}/system/ge_sph")
ADD_DEFINITIONS(-DCL_SPH_UTIL_SOURCE_DIR="${rtpslib_SOURCE_DIR}/system/utils")

INCLUDE_DIRECTORIES(${rtpslib_INCLUDES})
ADD_LIBRARY(rtps STATIC ${rtpslib_SOURCES})

TARGET_LINK_LIBRARIES (rtps
    #${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)

#ADD_SUBDIRECTORY(gordon_erle/datastructures_cl)
ADD_SUBDIRECTORY(system/utils)
