CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(enjacl)

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${enjacl_SOURCE_DIR}/../cmake
     )

message("${CMAKE_MODULE_PATH}\n")


FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

INCLUDE_DIRECTORIES( 
	${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    #${OPENCL_INCLUDE_DIR}
    ${enjacl_SOURCE_DIR}/../opencl10
)

#make OpenCL use the OpenGL context to avoid memory transfer
ADD_DEFINITIONS(-DGL_INTEROP)
ADD_DEFINITIONS(-DOPENCL_SHARED)
ADD_DEFINITIONS(-g )

#particles opencl library
set (CLCXXFILES  
    enja.cpp 
    opencl.cpp 
    physics/collision.cpp 
    render.cpp 
    util.cpp 
    timege.cpp)

SET(ocllib enjaclparticles)
ADD_LIBRARY(${ocllib} ${CLCXXFILES})


SET(EXEC_CL sph.x)
ADD_EXECUTABLE(${EXEC_CL} main.cpp util.cpp)


TARGET_LINK_LIBRARIES (${EXEC_CL}
   ${ocllib}
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)


