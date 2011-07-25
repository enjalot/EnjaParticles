IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(rtpslib C CXX)

#change this later
SET( CMAKE_MODULE_PATH ${RTPS_CMAKE_MODULE_PATH})
message("cmake_module_path: ${CMAKE_MODULE_PATH}\n")

#this mostly affects the path to cl files
SET(RTPS_DEBUG FALSE)
#SET(RTPS_DEBUG FALSE)

#only works on mac/linux
SET(PREPROCESS_CL TRUE)

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
ADD_DEFINITIONS(-DSIMPLE_CL_SOURCE_DIR="cl_simple")
ADD_DEFINITIONS(-DSPH_CL_SOURCE_DIR="cl_sph")
ADD_DEFINITIONS(-DFLOCK_CL_SOURCE_DIR="cl_flock")
ADD_DEFINITIONS(-DOUTER_CL_SOURCE_DIR="cl_outer")
ADD_DEFINITIONS(-DCOMMON_CL_SOURCE_DIR="cl_common")
ADD_DEFINITIONS(-DGLSL_BIN_DIR="shaders")
ADD_DEFINITIONS(-DGLSL_SOURCE_DIR="render")
##############################################################################


FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

message("OPENCL INCLUDE: ${OPENCL_INCLUDE_DIR}\n")

set (rtpslib_INCLUDES
    #${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    #${BASE_DIR}/opencl10
    ${OPENCL_INCLUDE_DIR}
    ${rtpslib_SOURCE_DIR}
    ${rtpslib_SOURCE_DIR}/system
    ${rtpslib_SOURCE_DIR}/system/sph
    ${rtpslib_SOURCE_DIR}/system/flock
    ${rtpslib_SOURCE_DIR}/system/outer
    ${rtpslib_SOURCE_DIR}/system/common
    ${rtpslib_SOURCE_DIR}/render
    ${rtpslib_SOURCE_DIR}/render/util
    ${rtpslib_SOURCE_DIR}/domain
    ${rtpslib_SOURCE_DIR}/opencl
    ${rtpslib_SOURCE_DIR}/matrix_library
    #${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src
)    

IF(CLOUD_COLLISION)
    SET(cloud_sources 
    ${rtpslib_SOURCE_DIR}/system/Cloud.cpp

    ${rtpslib_SOURCE_DIR}/system/sph/CloudVelocity
    ${rtpslib_SOURCE_DIR}/system/sph/CloudEuler
    ${rtpslib_SOURCE_DIR}/system/sph/Collision_cloud
    ${rtpslib_SOURCE_DIR}/system/common/CloudPermute
    )
ENDIF(CLOUD_COLLISION)


#library source files (don't explicitly give file extension, inferred from source type)
set (rtpslib_SOURCES
    ${rtpslib_SOURCE_DIR}/RTPS
    ${rtpslib_SOURCE_DIR}/RTPSettings
    ${rtpslib_SOURCE_DIR}/util
    ${rtpslib_SOURCE_DIR}/structs
    ${rtpslib_SOURCE_DIR}/timege    #gordon's timer class
    ${rtpslib_SOURCE_DIR}/timer_eb  #evan's timer class
    ${rtpslib_SOURCE_DIR}/rtps_common

    ${rtpslib_SOURCE_DIR}/render/Render
    ${rtpslib_SOURCE_DIR}/render/SpriteRender
    ${rtpslib_SOURCE_DIR}/render/SSFRender
    ${rtpslib_SOURCE_DIR}/render/Sphere3DRender
    ${rtpslib_SOURCE_DIR}/render/util/stb_image.c
    ${rtpslib_SOURCE_DIR}/render/util/stb_image_write.h

    ${rtpslib_SOURCE_DIR}/domain/Domain
    ${rtpslib_SOURCE_DIR}/domain/IV
    # ${rtpslib_SOURCE_DIR}/system/common/Hose
    

    # common use for all systems
    ${rtpslib_SOURCE_DIR}/system/common/Hose
    ${rtpslib_SOURCE_DIR}/system/common/Hash
    ${rtpslib_SOURCE_DIR}/system/common/CellIndices
    ${rtpslib_SOURCE_DIR}/system/common/Permute
	
    
    # Simple
    ${rtpslib_SOURCE_DIR}/system/Simple.cpp
    ${rtpslib_SOURCE_DIR}/system/simple/ForceField
    ${rtpslib_SOURCE_DIR}/system/simple/Euler
    	
	# SPH
    ${rtpslib_SOURCE_DIR}/system/SPH.cpp
    ${rtpslib_SOURCE_DIR}/system/SPHSettings.cpp

    #cloud
    ${cloud_sources}


    ${rtpslib_SOURCE_DIR}/system/sph/Euler
    ${rtpslib_SOURCE_DIR}/system/sph/LeapFrog
    ${rtpslib_SOURCE_DIR}/system/sph/Lifetime
    #${rtpslib_SOURCE_DIR}/system/sph/Prep
    #${rtpslib_SOURCE_DIR}/system/sph/Sort
    #${rtpslib_SOURCE_DIR}/system/sph/CloudSort
    ${rtpslib_SOURCE_DIR}/system/sph/Density
    ${rtpslib_SOURCE_DIR}/system/sph/Force
    ${rtpslib_SOURCE_DIR}/system/sph/Collision_wall
    ${rtpslib_SOURCE_DIR}/system/sph/Collision_triangle

    
    #### these are CPU only
    ${rtpslib_SOURCE_DIR}/system/sph/Pressure
    ${rtpslib_SOURCE_DIR}/system/sph/Viscosity
    ${rtpslib_SOURCE_DIR}/system/sph/XSPH
    ####


	# FLOCK
    ${rtpslib_SOURCE_DIR}/system/FLOCK.cpp
    ${rtpslib_SOURCE_DIR}/system/FLOCKSettings.cpp
    ${rtpslib_SOURCE_DIR}/system/flock/EulerIntegration
    ${rtpslib_SOURCE_DIR}/system/flock/Rules
    #${rtpslib_SOURCE_DIR}/system/flock/ComputeRules
    #${rtpslib_SOURCE_DIR}/system/flock/AverageRules
    #${rtpslib_SOURCE_DIR}/system/flock/LeapFrog
    #${rtpslib_SOURCE_DIR}/system/flock/Density
    #${rtpslib_SOURCE_DIR}/system/flock/Pressure
    #${rtpslib_SOURCE_DIR}/system/flock/Viscosity
    #${rtpslib_SOURCE_DIR}/system/flock/XFLOCK
    #${rtpslib_SOURCE_DIR}/system/flock/Collision_wall
    #${rtpslib_SOURCE_DIR}/system/flock/Collision_triangle

    # Added by GE
    ${rtpslib_SOURCE_DIR}/system/boids
	# OUTER
    ${rtpslib_SOURCE_DIR}/system/OUTER.cpp
    ${rtpslib_SOURCE_DIR}/system/OUTERSettings.cpp
    ${rtpslib_SOURCE_DIR}/system/outer/Euler
    ${rtpslib_SOURCE_DIR}/system/outer/LeapFrog
    ${rtpslib_SOURCE_DIR}/system/outer/Lifetime
    ${rtpslib_SOURCE_DIR}/system/outer/Prep
    ${rtpslib_SOURCE_DIR}/system/outer/Sort
    ${rtpslib_SOURCE_DIR}/system/outer/Density
    ${rtpslib_SOURCE_DIR}/system/outer/Force
    ${rtpslib_SOURCE_DIR}/system/outer/Collision_wall
    ${rtpslib_SOURCE_DIR}/system/outer/Collision_triangle
    #### these are CPU only
    ${rtpslib_SOURCE_DIR}/system/outer/Pressure
    ${rtpslib_SOURCE_DIR}/system/outer/Viscosity
    ${rtpslib_SOURCE_DIR}/system/OUTER
    ####

    
    # neighbors
    #${rtpslib_SOURCE_DIR}/system/flock/Prep
    ${rtpslib_SOURCE_DIR}/system/flock/Sort
    #${rtpslib_SOURCE_DIR}/system/flock/Hash
    #${rtpslib_SOURCE_DIR}/system/flock/BitonicSort
    #${rtpslib_SOURCE_DIR}/system/flock/DataStructures
    #${rtpslib_SOURCE_DIR}/system/flock/CloudDataStructures
    #${rtpslib_SOURCE_DIR}/system/flock/Neighbors
    #${rtpslib_SOURCE_DIR}/particle
    )

IF (WIN32)
    set (rtpslib_SOURCES 
    	${rtpslib_SOURCES}
    	${rtpslib_SOURCE_DIR}/gtod_windows
    )
ENDIF ()


IF(WITH_OPENCL)
SET(rtpslib_SOURCES ${rtpslib_SOURCES}
    ${rtpslib_SOURCE_DIR}/opencl/CLL
    ${rtpslib_SOURCE_DIR}/opencl/Kernel
    #${rtpslib_SOURCE_DIR}/opencl/Buffer #template class

    #${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src/oclBitonicSort_launcher
    #${rtpslib_SOURCE_DIR}/opencl/bitonic_sort/src/BitonicSort #template class
    #these should probably be in the OpenCL classes
    #${rtpslib_SOURCE_DIR}/system/sph/Scopy
    #${rtpslib_SOURCE_DIR}/system/flock/Scopy
    #${rtpslib_SOURCE_DIR}/system/sph/SetInt
    )
ENDIF(WITH_OPENCL)


INCLUDE_DIRECTORIES(${rtpslib_INCLUDES})
#ADD_LIBRARY(rtps SHARED ${rtpslib_SOURCES})
ADD_LIBRARY(rtps STATIC ${rtpslib_SOURCES})

TARGET_LINK_LIBRARIES (rtps
    #${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
   # /opt/local/lib/libGLEW.a #for static linking when distributing on mac. should do this more cmake way
   #/usr/lib/libGLEW.a #for static linking when distributing on linux. should do this more cmake way
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



ADD_SUBDIRECTORY(system/common)
ADD_SUBDIRECTORY(system/simple)
ADD_SUBDIRECTORY(system/sph)
ADD_SUBDIRECTORY(system/flock)
ADD_SUBDIRECTORY(system/outer)

MAKE_DIRECTORY(${rtpslib_BINARY_DIR}/shaders)
FILE(COPY ${rtpslib_SOURCE_DIR}/render/shaders/ DESTINATION ${rtpslib_BINARY_DIR}/shaders/)

#IF(WIN32)
install (TARGETS rtps DESTINATION ${rtpsuite_BINARY_DIR}/bin)
#install (TARGETS rtps DESTINATION ${RTPS_INSTALL_DIR})
#this should probably be done in a different way
INSTALL(CODE "FILE(MAKE_DIRECTORY ${RTPS_INSTALL_DIR}/shaders)")
install (DIRECTORY ${rtpslib_BINARY_DIR}/shaders/ DESTINATION ${RTPS_INSTALL_DIR}/shaders/)
#ENDIF(WIN32)
