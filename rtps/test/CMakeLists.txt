IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(test CXX)

#change this later
SET( BASE_DIR ${test_SOURCE_DIR}/../..)
SET( RTPS_DIR ${test_SOURCE_DIR}/../rtpslib)

message("base dir: ${BASE_DIR}")

#need to include FindOPENCL.cmake to
SET( CMAKE_MODULE_PATH
     ${BASE_DIR}/cmake
     )

message("${CMAKE_MODULE_PATH}\n")

FIND_PACKAGE (GLUT)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OPENCL)
FIND_PACKAGE (GLEW)

set (test_INCLUDES
    ${RTPS_DIR}
    ${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${BASE_DIR}/opencl10
)


#library
set (test_SOURCES
    main.cpp
    )



ADD_DEFINITIONS(-g )

INCLUDE_DIRECTORIES(${test_INCLUDES})
ADD_EXECUTABLE(test.x ${test_SOURCES})

TARGET_LINK_LIBRARIES (test.x
    rtps
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${GLEW_LIBRARY}
)


