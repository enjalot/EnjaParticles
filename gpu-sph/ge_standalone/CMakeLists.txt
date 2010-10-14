CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CXXFILES ConsoleTest.cpp)
SET(CUFILES ConsoleTest.cu)

INCLUDE_DIRECTORIES( 
	${OPENGL_INCLUDE_DIR}
    $ENV{CUDA_LOCAL}/../shared/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../SPHSimLib/ 
)

LINK_DIRECTORIES(/usr/local/cuda/lib /Developer/GPU_Computing/C/lib $ENV{CUDA_LOCAL}/common/lib /Developer/GPU_Computing/shared/lib/darwin)
CUDA_ADD_EXECUTABLE(sph ${CXXFILES} ${CUFILES})

# cuda libraries already in sph_lib
TARGET_LINK_LIBRARIES(sph 
   sph_lib
)
