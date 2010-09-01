CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(isph)

# compile for 64 bits

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${isph_SOURCE_DIR}/cmake
     )

message("${CMAKE_MODULE_PATH}\n")

ADD_DEFINITIONS(-DOPENCL_SHARED)


FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)


INCLUDE_DIRECTORIES( 
	${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
)

#provide the source path so we can open our .cl file at runtime
#make OpenCL use the OpenGL context to avoid memory transfer
ADD_DEFINITIONS(-DGL_INTEROP)
ADD_DEFINITIONS(-g )



ADD_SUBDIRECTORY(isphlib)
ADD_SUBDIRECTORY(tools/test)
