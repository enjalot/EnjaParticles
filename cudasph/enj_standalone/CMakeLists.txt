#CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#project(enjacl)

#ADD_DEFINITIONS(-m32)

#need to include FindOPENCL.cmake to
#SET( CMAKE_MODULE_PATH
#     ${enjacl_SOURCE_DIR}/../cmake
#     )

message("${CMAKE_MODULE_PATH}\n")

ADD_DEFINITIONS(-DOPENCL_SHARED)


#FIND_PACKAGE (GLUT)
#FIND_PACKAGE (OpenGL)
#FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

#SET(GUTILS $ENV{GRAPHICS_LIBS}/graphics_utils)
#SET(GRAND $ENV{GRAPHICS_LIBS}/random)

#SET(GUTILS_INC $ENV{GRAPHICS_LIBS_SRC}/graphics_utils)
#SET(GRAND_INC $ENV{GRAPHICS_LIBS_SRC}/random)

#message("MYCUDAINCLUDES: ${MY_CUDA_INCLUDES}\n")

INCLUDE_DIRECTORIES( 
#	${GRAND_INC}
#	${GUTILS_INC}
	${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    #${OPENCL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../opencl10
    
    $ENV{CUDA_LOCAL}/../shared/inc
	${MY_CUDA_INCLUDES}
    ${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/../SPHSimLib

)

LINK_DIRECTORIES(
     ${CMAKE_CURRENT_BINARY_DIR}/SPHSimLib
     $ENV{CUDA_LOCAL}/lib
	 $ENV{CUDA_LOCAL}/common/lib 
	 $ENV{CUDA_LOCAL}/../shared/lib/darwin 
	 $ENV{CUDA_LOCAL}/../shared/lib/linux
     $ENV{CUDA_HOME}/lib
	 $ENV{CUDA_HOME}/lib64
)


message("\nenjacl INCLUDE_DIRECTORIES\n")
get_directory_property(includes INCLUDE_DIRECTORIES)
message("${includes}\n")

#provide the source path so we can open our .cl file at runtime
#ADD_DEFINITIONS(-DCL_SOURCE_DIR="${enjacl_SOURCE_DIR}")
#make OpenCL use the OpenGL context to avoid memory transfer
ADD_DEFINITIONS(-DGL_INTEROP)
ADD_DEFINITIONS(-g )

#particles opencl library
set (CLCXXFILES  enja.cpp opencl.cpp physics/collision.cpp render.cpp 
    util.cpp timege.cpp 
    EnjaCudaHelper.cpp EnjaSimBuffer.cpp
)

SET(ocllib enjaclparticles)
ADD_LIBRARY(${ocllib} ${CLCXXFILES})


SET(EXEC_CL enjsph.x)
ADD_EXECUTABLE(${EXEC_CL} main.cpp util.cpp)


TARGET_LINK_LIBRARIES (${EXEC_CL}
#   ${GRAND}/librandom.a
#   ${GUTILS}/libgraphic_utilities.a
   ${ocllib}
    sph_lib
	sph_cuda
	sph_lib
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${MY_CUDA_LIBRARIES}
   ${GLEW_LIBRARY}
)


#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.v.glsl ${CMAKE_CURRENT_BINARY_DIR}/enja.v.glsl)
#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.f.glsl ${CMAKE_CURRENT_BINARY_DIR}/enja.f.glsl)
#don't need to copy it if we provide path to source dir
#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/enja.cl ${CMAKE_CURRENT_BINARY_DIR}/enja.cl)

