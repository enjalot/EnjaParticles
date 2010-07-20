CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(enjacl)

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${enjacl_SOURCE_DIR}/../cmake)

message("${CMAKE_MODULE_PATH}\n")


FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)
FIND_PACKAGE (OpenCV REQUIRED)

#message("OpenCV stuff\n ${OpenCV_LIB_DIR} \n ${OpenCV_LIBS}\n ${OpenCV_INCLUDE_DIRS}\n")

INCLUDE_DIRECTORIES( 
	${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${OPENCL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

message("\nenjacl INCLUDE_DIRECTORIES\n")
get_directory_property(includes INCLUDE_DIRECTORIES)
message("${includes}\n")

#provide the source path so we can open our .cl file at runtime
ADD_DEFINITIONS(-DCL_SOURCE_DIR="${enjacl_SOURCE_DIR}")
#make OpenCL use the OpenGL context to avoid memory transfer
ADD_DEFINITIONS(-DGL_INTEROP)
ADD_DEFINITIONS(-g )

#particles opencl library
set (CLCXXFILES  enja.cpp opencl.cpp render.cpp util.cpp timege.cpp)
SET(ocllib enjaclparticles)
ADD_LIBRARY(${ocllib} ${CLCXXFILES})


SET(EXEC_CL picture.x)
ADD_EXECUTABLE(${EXEC_CL} main_picture.cpp util.cpp)

TARGET_LINK_LIBRARIES (${EXEC_CL}
   ${ocllib}
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
   ${OpenCV_LIBS}
)


#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.v.glsl ${CMAKE_CURRENT_BINARY_DIR}/enja.v.glsl)
#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.f.glsl ${CMAKE_CURRENT_BINARY_DIR}/enja.f.glsl)
#don't need to copy it if we provide path to source dir
#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.cl ${CMAKE_CURRENT_BINARY_DIR}/enja.cl)

