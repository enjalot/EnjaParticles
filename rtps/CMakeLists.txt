IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(rtpsuite)

ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/rtpslib)
ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/test)

