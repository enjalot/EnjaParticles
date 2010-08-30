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


#Doxygen
find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT   ${isph_SOURCE_DIR}/doc/doxyfile)
set(DOXYGEN_OUTPUT  ${isph_BINARY_DIR}/html/index.html)
file(MAKE_DIRECTORY ${isph_BINARY_DIR}/html)

configure_file(
  ${isph_SOURCE_DIR}/doc/doxyfile
  ${DOXYGEN_INPUT} @ONLY
)

add_custom_command(
  OUTPUT  ${DOXYGEN_OUTPUT}
  COMMAND ${DOXYGEN} ${DOXYGEN_INPUT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DOXYGEN_INPUT}
)

add_custom_target(doc DEPENDS ${DOXYGEN_OUTPUT})




ADD_SUBDIRECTORY(isphlib)
ADD_SUBDIRECTORY(tools/test)
