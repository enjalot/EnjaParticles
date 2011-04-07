IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(rtpsuite)


#need to include FindOPENCL.cmake to
SET( RTPS_CMAKE_MODULE_PATH
     ${rtpsuite_SOURCE_DIR}/cmake
)
SET( OPENCL_INCLUDE_DIR 
    ${rtpsuite_SOURCE_DIR}/opencl10)

message("rtps_cmake_module_path: ${RTPS_CMAKE_MODULE_PATH}\n")


ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/rtpslib)
ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/test)

