# C is not required on the mac, but appears to be on linux
project(cudasph C CXX)

IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)



SET( CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
	$ENV{CMAKE_MODULES}
)


SET(CUDA_64_BIT_DEVICE_CODE ON)
message("IN CMAKELISTS, 64bit ON?: ${CUDA_64_BIT_DEVICE_CODE}\n")
# Get common build instructions used by all operating systems
INCLUDE (COMMON_CONFIG)
INCLUDE (CUDA_CONFIG)

ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/SPHSimLib/)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/ConsoleTest/)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/enj_standalone/)

