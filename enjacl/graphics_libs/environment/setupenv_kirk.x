
# setup environment variables for kirk

export BASE_DIR=${HOME}/vislab/src/heptadiagonal/gravimetrie_roland/branch_grav/branch1/graphics_libs

export CUDA_HOME=$HOME/vislab/NVIDIA_CUDA_TOOLKIT_3.0
export CUDA_LOCAL=$HOME/vislab/NVIDIA_CUDA_SDK_3.0/C
export OPENCL_HOME=${HOME}/vislab/NVIDIA_CUDA_SDK_3.0/OpenCL
export GRAPHIC_LIBS_HOME=${BASE_DIR}

#export CUDA_PROFILE_CONFIG=./profile_config

#export SWAN=${BASE_DIR}/swan2
#export PATH=${SWAN}/bin:${PATH}
#export LD_LIBRARY_PATH=${SWAN}/lib:${CUDA_HOME}/lib64:/usr/lib64:/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${SWAN}/lib:${CUDA_HOME}/lib:/usr/lib32:/lib32
#:${LD_LIBRARY_PATH}


export CXX=/usr/bin/g++-4.3
export CC=/usr/bin/gcc-4.3

# -fpermissive solves some problems related to non-declaration of printf
#export CXX=/usr/bin/g++-4.4 
#export CC=/usr/bin/gcc-4.4 
#export CXXFLAGS=-fpermissive
#export CCFLAGS=-fpermissive

export FC=/usr/bin/gfortran

export CMAKE_MODULES=/usr/share/cmake-2.8/Modules/

unset CUDA_INSTALL_PATH
unset CUDA_LIBS

export CUDA_LIB_PATH=/usr/lib32
export CUDART_PATH=${CUDA_HOME}/lib
export CUBLAS_PATH=${CUDA_HOME}/lib

