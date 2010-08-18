CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CXXFILES 
	SimBufferCuda.cpp
	SimCudaHelper.cpp
	#SimMemoryManager.cpp
	SimSettings.cpp
	SimpleSPH.cpp
	SimulationSystem.cpp
)

SET(CUFILES 
	CudaMax.cu
	CudaUtils.cu
#	K_Boundaries_Common.cu
#	K_Boundaries_Terrain.cu
#	K_Boundaries_Walls.cu
#	#K_SPH_Kernels.cu
#	#K_SPH_Kernels_cubic.cu
#	#K_SPH_Kernels_gaussian.cu
#	#K_SPH_Kernels_poly6.cu
#	#K_SPH_Kernels_quadratic.cu
#	#K_SPH_Kernels_quartic.cu
#	#K_SPH_Kernels_quintic.cu
#	#...
#	#...
#	#K_SPH_Kernels_spiky.cu
#	#K_SPH_Kernels_viscosity.cu
#	#GE K_SimDEM.cu
#	K_SimpleSPH.cu
#	K_SimpleSPH_Integrate.cu
#	K_SimpleSPH_Step1.cu
#	K_SimpleSPH_Step2.cu
#	#K_SnowSPH.cu
#	#K_SnowSPH_Integrate.cu
#	#K_SnowSPH_Step1.cu
#	#K_SnowSPH_Step2.cu
#	#K_SnowSPH_Step3.cu
#	#K_UniformGrid_Hash.cu
#	K_UniformGrid_NeighborList.cu   # should work
#	#K_UniformGrid_Update.cu
#	#K_UniformGrid_Utils.cu
	SimBase.cu
	SimDEM.cu
	SimSimpleSPH.cu  #
	SimSnowSPH.cu
	UniformGrid.cu
	cuPrintf.cu
#	radixsort.cu  # in the libcudpp library
	srts_radix_sort.cu
##	srts_radix_sort_kernel.cu
#	srts_test_radix_sort.cu
	srts_verifier.cu
	timer.cu
	checkCudaError.cu  # added by GE
)


# Problem: cmake . does not work unless libraries already exist. But 
# they will get created when I execute Makefile, before getting to this point


IF(APPLE)
	set (MYOPENCL_LIBRARIES32 ${OPENCL_LIBRARIES})
ELSE(APPLE)
	set (MYOPENCL_LIBRARIES32 /usr/lib32/libOpenCL.so)
ENDIF(APPLE)


# IMPORTANT: Order of libraries is important under linux!!!
# Make sure that gpu comes before ${MY_LIBRARIES}!!!

INCLUDE_DIRECTORIES(
	.
	/Developer/GPU_Computing/shared/inc/
	${OPENGL_INCLUDE_DIR}
)

CUDA_INCLUDE_DIRECTORIES(
	  ${MY_CUDA_INCLUDES}
	  ${OPENGL_INCLUDE_DIR}
      ${CUDA_SDK_ROOT_DIR}/common/inc
	  ${CUDA_TOOLKIT_ROOT_DIR}/include
)

CUDA_ADD_LIBRARY(sph_cuda ${CUFILES})
#TARGET_LINK_LIBRARIES(sph_cuda)

LINK_DIRECTORIES(/usr/local/cuda/lib /Developer/GPU_Computing/C/lib $ENV{CUDA_LOCAL}/common/lib /Developer/GPU_Computing/shared/lib/darwin)
ADD_LIBRARY(sph_lib ${CXXFILES})

# ON mac

TARGET_LINK_LIBRARIES(sph_lib
	/Developer/GPU_Computing/C/common/lib/darwin/libcudpp_i386.a
	sph_cuda
	#${MY_LIBRARIES}
	/Developer/GPU_Computing/C/lib/libcutil_i386.a
	#/Users/erlebach/Documents/src/gpu_sph_sim_source_code/gpusphsim-read-only/SPHSimLib/libcudpp_i386.a   # removed radixSort to avoid conflict
	#/usr/local/cuda/lib/libcublas.dylib
	#/usr/local/cuda/lib/libcuda.dylib
	#/usr/local/cuda/lib/libcudart.dylib
	${MY_CUDA_LIBRARIES}
	${OPENGL_LIBRARIES}
)

message("*** MY_CUDA_LIBRARIES: ${MY_CUDA_LIBRARIES} ***")

