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

#FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

set (rtpslib_INCLUDES
    #${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${BASE_DIR}/opencl10
)


#library
set (rtpslib_SOURCES
    ${rtpslib_SOURCE_DIR}/RTPS.cpp
    ${rtpslib_SOURCE_DIR}/RTPSettings.cpp
    ${rtpslib_SOURCE_DIR}/util.cpp

    ${rtpslib_SOURCE_DIR}/opencl/CLL.cpp
    ${rtpslib_SOURCE_DIR}/opencl/Kernel.cpp
    #${rtpslib_SOURCE_DIR}/opencl/Buffer.cpp

    ${rtpslib_SOURCE_DIR}/render/Render.cpp

    ${rtpslib_SOURCE_DIR}/particle/UniformGrid.cpp
    
    ${rtpslib_SOURCE_DIR}/system/Simple.cpp
    
    ${rtpslib_SOURCE_DIR}/system/SPH.cpp
    ${rtpslib_SOURCE_DIR}/system/sph/Euler.cpp
    ${rtpslib_SOURCE_DIR}/system/sph/Density.cpp
    ${rtpslib_SOURCE_DIR}/system/sph/Pressure.cpp
    ${rtpslib_SOURCE_DIR}/system/sph/Collision_wall.cpp


    #${rtpslib_SOURCE_DIR}/particle
    )



ADD_DEFINITIONS(-g )

INCLUDE_DIRECTORIES(${rtpslib_INCLUDES})
ADD_LIBRARY(rtps STATIC ${rtpslib_SOURCES})

TARGET_LINK_LIBRARIES (rtps
    #${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)


