CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#project(graphics_library_examples)

ADD_DEFINITIONS(-g)

INCLUDE_DIRECTORIES( BEFORE
	${OPENCL_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR}
)

#ADD_SUBDIRECTORY(add_arrays)
ADD_SUBDIRECTORY(cublas_dble)

GET_DIRECTORY_PROPERTY(incs DIRECTORY cublas_dble INCLUDE_DIRECTORIES)
message("include directories from cublas: ${incs}")
