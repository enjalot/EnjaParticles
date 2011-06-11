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

IF(RTPS_INSTALL_DIR)
    message("**rtps install dir: ${RTPS_INSTALL_DIR}")
    message("**rtps install dir relative: ${RTPS_INSTALL_DIR_REL}")
ELSE(RTPS_INSTALL_DIR)
    set(RTPS_INSTALL_DIR ${rtpsuite_BINARY_DIR}/bin)
    set(RTPS_INSTALL_DIR_REL .)
ENDIF(RTPS_INSTALL_DIR)

message("rtps install dir: ${RTPS_INSTALL_DIR}")
message("rtps install dir relative: ${RTPS_INSTALL_DIR_REL}")
#INSTALL(CODE "FILE(MAKE_DIRECTORY \${ENV}\${CMAKE_INSTALL_PREFIX}${mydir})")
INSTALL(CODE "FILE(MAKE_DIRECTORY ${RTPS_INSTALL_DIR})")
#INSTALL(CODE "message(\"hmm: ${RTPS_INSTALL_DIR}\")")

ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/rtpslib)
ADD_SUBDIRECTORY (${rtpsuite_SOURCE_DIR}/test)

