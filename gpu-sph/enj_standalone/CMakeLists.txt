cmake_minimum_required(VERSION 2.8)
project(enjstandalone)

SET( CMAKE_MODULE_PATH
     ${enjstandalone_SOURCE_DIR}/../../cmake
)


FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)




SET(CXXFILES 
	EnjaCudaHelper.cpp
	EnjaSimBuffer.cpp
)

INCLUDE_DIRECTORIES(
	  $ENV{CUDA_LOCAL}/../shared/inc
	  ${MY_CUDA_INCLUDES}
      ${OPENGL_INCLUDE_DIR}
      ${CUDA_SDK_ROOT_DIR}/common/inc
      $ENV{CUDA_LOCAL}/common/inc
	  ${CUDA_TOOLKIT_ROOT_DIR}/include
      $ENV{CUDA_HOME}/include
	  ${CMAKE_CURRENT_SOURCE_DIR}/../SPHSimLib
	  ${CMAKE_CURRENT_SOURCE_DIR}/../../enjacl
	  ${CMAKE_CURRENT_SOURCE_DIR}/../../opencl10
)

ADD_LIBRARY(enja ${CXXFILES})


LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../enjacl/build)

message("glew lib: ${GLEW_LIBRARY}")

ADD_EXECUTABLE(sph_enja 
   main.cpp
)

TARGET_LINK_LIBRARIES(sph_enja
	enja
	enjaclparticles     # make sure library is in source directory
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${GLEW_LIBRARY}
)

