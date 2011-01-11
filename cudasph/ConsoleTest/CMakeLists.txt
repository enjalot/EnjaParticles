CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CXXFILES ConsoleTest.cpp)
#SET(CUFILES ConsoleTest.cu)

INCLUDE_DIRECTORIES( 
	${OPENGL_INCLUDE_DIR}
    #$ENV{CUDA_LOCAL}/../shared/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../SPHSimLib/ 
    ${CUDA_SDK_ROOT_DIR}/../shared/inc
	${MY_CUDA_INCLUDES}
    ${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_ROOT_DIR}/include

)

LINK_DIRECTORIES(
     $ENV{CUDA_LOCAL}/lib 
	 $ENV{CUDA_LOCAL}/common/lib 
	 $ENV{CUDA_LOCAL}/../shared/lib/darwin 
	 $ENV{CUDA_LOCAL}/../shared/lib/linux
	 $ENV{CUDA_HOME}/lib
)


message("CUDA>>>> ${MY_CUDA_LIBRARIES}")
#CUDA_ADD_EXECUTABLE(sph ${CXXFILES} ${CUFILES})
ADD_EXECUTABLE(console ${CXXFILES})

#ADD_DEPENDENCIES(sph ${cudpp})

# cuda libraries already in sph_lib
TARGET_LINK_LIBRARIES(console
	sph_lib
    sph_cuda
    sph_lib  # repeat for linux
	${cudpp}
    #$ENV{CUDA_LOCAL}/lib/libcutil_i386.a
	${MY_CUDA_LIBRARIES}
	${OPENGL_LIBRARIES}
)
