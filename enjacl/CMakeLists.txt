CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(enjacl)

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${enjacl_SOURCE_DIR}/../cmake
     )

message("${CMAKE_MODULE_PATH}\n")

#Only define this on the apple platform
# appears in RadixSort.cl
IF(APPLE)
	ADD_DEFINITIONS(-DMAC)
ENDIF()
ADD_DEFINITIONS(-DCL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/physics")
#ADD_DEFINITIONS(-DCL_SOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}")


FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

INCLUDE_DIRECTORIES( 
	oclRadixSort/inc
	${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
	$ENV{OPENCL_HOME}/common/inc
	$ENV{OPENCL_HOME}/../shared/inc
    #${OPENCL_INCLUDE_DIR}
    ${enjacl_SOURCE_DIR}/../opencl11
)

message("Opencl Home: $ENV{OPENCL_HOME}")

#make OpenCL use the OpenGL context to avoid memory transfer
ADD_DEFINITIONS(-DGL_INTEROP)
ADD_DEFINITIONS(-DOPENCL_SHARED)
ADD_DEFINITIONS(-g )

#particles opencl library
set (CLCXXFILES  
	oclRadixSort/src/RadixSort.cpp
	oclRadixSort/src/Scan.cpp
    enja.cpp 
    opencl.cpp 
    physics/collision.cpp 
    render.cpp 

    SPH.cpp

    util.cpp 
    timege.cpp)

SET(ocllib enjaclparticles)
ADD_LIBRARY(${ocllib} ${CLCXXFILES})


SET(EXEC_CL sph.x)
ADD_EXECUTABLE(${EXEC_CL} main.cpp util.cpp)

TARGET_LINK_LIBRARIES (${ocllib}
   $ENV{OPENCL_HOME}/../shared/lib/libshrutil_x86_64.a
   $ENV{OPENCL_HOME}/common/lib/liboclUtil_x86_64.a
)

TARGET_LINK_LIBRARIES (${EXEC_CL}
   $ENV{OPENCL_HOME}/../shared/lib/libshrutil_x86_64.a
   $ENV{OPENCL_HOME}/common/lib/liboclUtil_x86_64.a
   ${ocllib}
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)


