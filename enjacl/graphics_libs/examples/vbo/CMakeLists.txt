CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#project(gravimetrie Fortran)

# Must be set before add_executable (appends new directory)
#LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib /usr/lib32 )
LINK_DIRECTORIES( ${CUDA_TOOLKIT_ROOT_DIR}/lib /usr/lib32 )

#FIND_LIBRARY(GLEW_LIB GLEW PATHS ${CUDA_SDK_ROOT_DIR}/common/lib/darwin)
FIND_LIBRARY(GLEW_LIB GLEW PATHS /usr/lib )

# Problem: cmake . does not work unless libraries already exist. But 
# they will get created when I execute Makefile, before getting to this point

# previously called FIND_PACKAGE(GLUT)

SET(myex vbo.x)
ADD_EXECUTABLE(${myex} vbo.cpp)

TARGET_LINK_LIBRARIES (${myex}
   graphic_utilities
   utilities
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   #cuda
   #cudart
   ${GLEW_LIB}
   #${CUDA_CUT_LIBRARIES}
)
