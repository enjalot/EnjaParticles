
# setup environment variables for mac

export BASE_DIR=${HOME}/Documents/roland_martin/gravimetrie_roland/branch_grav/branch3/

export GRAPHIC_LIBS_HOME=${BASE_DIR}/graphics_libs

export CUDA_HOME=/usr/local/cuda
export CUDA_LOCAL=/Developer/GPU_Computing/C
export OPENCL_HOME=${CUDA_LOCAL}/../OpenCL

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

export DYLD_LIBRARY_PATH=${CUDA_LOCAL}/share/lib/darwin:${CUART_PATH}:${DYLD_LIBRARY_PATH}
