
# setup environment variables for mark2 (machine with fermi)

BASE_DIR=${HOME}/src/gravimetrie/branch_grav/branch1

#export SWAN=${HOME}/src/swan2_kirk
#export SWAN=${BASE_DIR}/swan2

export CUDA_HOME=/usr/local/cuda
export CUDA_LOCAL=/usr/local/NVIDIA_GPU_Computing_SDK/C
export OPENCL_HOME=/usr/local/NVIDIA_GPU_Computing_SDK/OpenCL
export GRAPHIC_LIBS_HOME=${BASE_DIR}

export CUDA_PROFILE_CONFIG=./profile_config


export PATH=${SWAN}/bin:${PATH}
# 64 bit
#export LD_LIBRARY_PATH=${SWAN}/lib:${CUDA_HOME}/lib64:/usr/lib64:/lib64
# 32 bit
export LD_LIBRARY_PATH=${SWAN}/lib:${CUDA_HOME}/lib:/usr/lib:/lib

export CXX=/usr/bin/g++-4.3
export CC=/usr/bin/gcc-4.3
export FC=/usr/bin/gfortran

export CMAKE_MODULES=/usr/local/share/cmake-2.8/Modules

unset CUDA_INSTALL_PATH
unset CUDA_LIBS

# 32 bit
export CUDA_LIB_PATH=/usr/lib
export CUDART_PATH=${CUDA_HOME}/lib
export CUBLAS_PATH=${CUDA_HOME}/lib

