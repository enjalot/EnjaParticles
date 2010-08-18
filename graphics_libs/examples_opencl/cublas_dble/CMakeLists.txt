CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

ADD_DEFINITIONS(-DOPENCL)

# Must be set before add_executable (appends new directory)
LINK_DIRECTORIES( ${CUDA_TOOLKIT_ROOT_DIR}/lib /usr/lib32 )

#FIND_LIBRARY(GLEW_LIB GLEW PATHS /usr/lib )

# Problem: cmake . does not work unless libraries already exist. But 
# they will get created when I execute Makefile, before getting to this point

# previously called FIND_PACKAGE(GLUT)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )


set (CXXFILES  sscal.cpp scopy.cpp saxpy.cpp sdot.cpp mat_mat_mul_el.cpp mat_vec.cpp inverse_diag_3d_kernel.cpp inv_test.cpp mat_vec_dirichlet.cpp )


SET(OCL_LIB ocl_common)
ADD_LIBRARY(${OCL_LIB} ${CXXFILES})

# used in C++ code to indicate location of OpenCL kernels
add_definitions(-DCL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")


SET(EXEC test_common_ocl.x)
ADD_EXECUTABLE(${EXEC} main.cpp )

TARGET_LINK_LIBRARIES (${EXEC}
   ${OCL_LIB}
   ${MY_LIBRARIES}
   #${GLUT_LIBRARIES}
   #${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
)

