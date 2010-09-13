# Environment variables
#	GRAPHIC_LIBS_HOME/  : subdirectories include graphic_utils, utilities, ...
#	CUDA_HOME : root of cuda installation: (e.g., /usr/local/cuda, under which
#	     one has bin/, ...)
#	CUDA_LOCAL : root of NVIDIA_SDK, under which one finds common/, lib/, #	bin/, ...


cmake_minimum_required(VERSION 2.6)

set(utils cu.cpp array_cuda_1d.cpp array_cuda_t.cpp array_cuda_1d_cu.cpp)

# Create a symbolic link swan/ to the actual swan/ installation. 
# Do not include this link in the svn repository!
# Create a symbolic link to the swan installation, but do not include 
# it in the svn repository

set (CXXFILES cu.cpp array_cuda_1d.cpp array_cuda_t.cpp array_cuda_1d_cu.cpp)

Include_directories( 
	${SWAN}/include
)

add_library(swan_utilities STATIC ${utils})

