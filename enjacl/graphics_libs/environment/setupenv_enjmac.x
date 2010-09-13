
# setup environment variables for kirk

BASE_DIR=${HOME}/code/graphics_libs

#export CUDA_HOME=/usr/local/cuda
export CUDA_HOME=/Developer/GPU\ Computing
export CUDA_LOCAL=/Developer/GPU\ Computing/C
export OPENCL_HOME=${CUDA_LOCAL}/../OpenCL
export GRAPHIC_LIBS_HOME=${BASE_DIR}

# the latest version of swan does not work on the mac
#export SWAN=${BASE_DIR}/swan2
#export SWAN=${BASE_DIR}/../../trunk/swan
export SWAN=${BASE_DIR}/swan
export PATH=$SWAN/bin:$PATH


#export CUDA_PROFILE_CONFIG=./profile_config

export DYLD_LIBRARY_PATH=${SWAN}/lib/:${DYLD_LIBRARY_PATH}:${CUDA_LOCAL}/common/lib/darwin
export FC=gfortran
export FCFLAGS=-m32

export CMAKE_MODULES=/opt/local/share/cmake-2.8/Modules/

export CUDA_LIB_PATH=/usr/lib32
export CUDART_PATH=${CUDA_HOME}/lib
export CUBLAS_PATH=${CUDA_HOME}/lib
