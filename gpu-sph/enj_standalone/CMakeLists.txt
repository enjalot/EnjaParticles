

SET(CXXFILES 
	EnjaCudaHelper.cpp
	EnjaSimBuffer.cpp
)

INCLUDE_DIRECTORIES(
	  $ENV{CUDA_LOCAL}/../shared/inc
	  ${MY_CUDA_INCLUDES}
	  ${OPENGL_INCLUDE_DIR}
      ${CUDA_SDK_ROOT_DIR}/common/inc
	  ${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_SOURCE_DIR}/SPHSimLib
)

ADD_LIBRARY(enja ${CXXFILES})

