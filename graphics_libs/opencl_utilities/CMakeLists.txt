CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)

set (CXXFILES cl.cpp array_opencl_t.cpp array_opencl_1d.cpp array_opencl_1d_cl.cpp cll_program.cpp cll_kernel.cpp print_dev_info.cpp device_info.cpp oclUtils.cpp shrUtils.cpp)

Include_directories( BEFORE
	${OPENCL_INCLUDE_DIR}
)

ADD_LIBRARY(cl_utilities_cpp STATIC ${CXXFILES})

