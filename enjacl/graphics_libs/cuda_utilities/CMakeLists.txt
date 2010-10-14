CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


set (CXXFILES cu.cpp array_cuda_1d.cpp array_cuda_t.cpp)
set (CUFILES  array_cuda_1d_cu.cu)


Include_directories( 
    . 
	# why is this line necessary? 
	# not necessary if cmake from branch1/build
	# necessary if cmake graphics_libs
    ${CMAKE_CURRENT_DIRECTORY}/../utilities   
	$ENV{CUDA_HOME}/include 
	$ENV{CUDA_LOCAL}/common/inc 
)


#ADD_DEFINITIONS(-m32) # C++, already added for nvcc
CUDA_ADD_LIBRARY(cuda_utilities_cu STATIC ${CUFILES})
ADD_LIBRARY(cuda_utilities_cpp STATIC ${CXXFILES})

