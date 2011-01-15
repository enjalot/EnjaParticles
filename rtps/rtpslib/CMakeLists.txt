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


###### SETTING RELATED TO GPU VS CPU IMPLEMENTATIONS ##########################
SET(WITH_OPENCL TRUE)

ADD_DEFINITIONS(-g )
#ADD_DEFINITIONS(-DDEBUG)
IF(WITH_OPENCL)
    ADD_DEFINITIONS(-DGPU)
ENDIF(WITH_OPENCL)
#might want to have CPU even if we have OpenCL available
#ADD_DEFINITIONS(-DCPU)

##############################################################################

##### Source Paths ############################################################
ADD_DEFINITIONS(-DSIMPLE_CL_SOURCE_DIR="${rtpslib_BINARY_DIR}/cl_simple")
ADD_DEFINITIONS(-DSPH_CL_SOURCE_DIR="${rtpslib_BINARY_DIR}/cl_sph")
ADD_DEFINITIONS(-DBITONIC_CL_SOURCE_DIR="${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/")
ADD_DEFINITIONS(-DGLSL_SOURCE_DIR="${rtpslib_SOURCE_DIR}/render/")



##############################################################################


FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)


set (rtpslib_INCLUDES
    #${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${BASE_DIR}/opencl10
    ${rtpslib_SOURCE_DIR}
    ${rtpslib_SOURCE_DIR}/system
    ${rtpslib_SOURCE_DIR}/render
    ${rtpslib_SOURCE_DIR}/domain
    ${rtpslib_SOURCE_DIR}/opencl
    ${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src
    
    ${OpenCV_INCLUDE_DIRS}

)


#library source files (don't explicitly give file extension, inferred from source type)
set (rtpslib_SOURCES
    ${rtpslib_SOURCE_DIR}/RTPS
    ${rtpslib_SOURCE_DIR}/RTPSettings
    ${rtpslib_SOURCE_DIR}/util
    ${rtpslib_SOURCE_DIR}/structs
    ${rtpslib_SOURCE_DIR}/timege

    ${rtpslib_SOURCE_DIR}/render/Render

    ${rtpslib_SOURCE_DIR}/domain/Domain
    ${rtpslib_SOURCE_DIR}/domain/IV

    
    ${rtpslib_SOURCE_DIR}/system/Simple.cpp
    ${rtpslib_SOURCE_DIR}/system/simple/ForceField
    ${rtpslib_SOURCE_DIR}/system/simple/Euler
    
    ${rtpslib_SOURCE_DIR}/system/SPH.cpp
    ${rtpslib_SOURCE_DIR}/system/sph/Euler
    ${rtpslib_SOURCE_DIR}/system/sph/LeapFrog
    ${rtpslib_SOURCE_DIR}/system/sph/Density
    ${rtpslib_SOURCE_DIR}/system/sph/Pressure
    ${rtpslib_SOURCE_DIR}/system/sph/Viscosity
    ${rtpslib_SOURCE_DIR}/system/sph/XSPH
    ${rtpslib_SOURCE_DIR}/system/sph/Collision_wall
    
    ${rtpslib_SOURCE_DIR}/system/sph/Prep
    ${rtpslib_SOURCE_DIR}/system/sph/Hash
    ${rtpslib_SOURCE_DIR}/system/sph/BitonicSort
    ${rtpslib_SOURCE_DIR}/system/sph/DataStructures
    ${rtpslib_SOURCE_DIR}/system/sph/Neighbors
	
    #${rtpslib_SOURCE_DIR}/particle
    )

IF(WITH_OPENCL)
SET(rtpslib_SOURCES ${rtpslib_SOURCES}
    ${rtpslib_SOURCE_DIR}/opencl/CLL
    ${rtpslib_SOURCE_DIR}/opencl/Kernel
    #${rtpslib_SOURCE_DIR}/opencl/Buffer #template class

    #${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src/oclBitonicSort_launcher
    #${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src/BitonicSort #template class
    #these should probably be in the OpenCL classes
    ${rtpslib_SOURCE_DIR}/system/sph/Scopy
    #${rtpslib_SOURCE_DIR}/system/sph/SetInt
    )
ENDIF(WITH_OPENCL)


INCLUDE_DIRECTORIES(${rtpslib_INCLUDES})
ADD_LIBRARY(rtps SHARED ${rtpslib_SOURCES})

TARGET_LINK_LIBRARIES (rtps
    #${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)


#Doxygen
#find_package(Doxygen REQUIRED)
find_package(Doxygen)
set(DOXYGEN_INPUT   ${rtpslib_SOURCE_DIR}/doc/doxyfile)
set(DOXYGEN_OUTPUT  ${rtpslib_BINARY_DIR}/html/index.html)
file(MAKE_DIRECTORY ${rtpslib_BINARY_DIR}/html)

configure_file(
  ${rtpslib_SOURCE_DIR}/doc/doxyfile
  ${DOXYGEN_INPUT} @ONLY
)

add_custom_command(
  OUTPUT  ${DOXYGEN_OUTPUT}
  COMMAND ${DOXYGEN} ${DOXYGEN_INPUT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DOXYGEN_INPUT}
)

add_custom_target(doc DEPENDS ${DOXYGEN_OUTPUT})



ADD_SUBDIRECTORY(system/simple)
ADD_SUBDIRECTORY(system/sph)
