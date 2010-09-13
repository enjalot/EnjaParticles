
message("${CMAKE_SOURCE_DIR}")
message("${CMAKE_CURRENT_SOURCE_DIR}")

IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source! Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
#ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})


cmake_minimum_required(VERSION 2.6)

project(graphics_libs)

OPTION (PURE_OPENCL "OpenCL code with no support from Cuda" ON)

if (${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_CURRENT_SOURCE_DIR})
	SET( CMAKE_MODULE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/common
		$ENV{CMAKE_MODULES})
		message("${CMAKE_MODULE_PATH}\n")
	INCLUDE(common/cmake_standalone.txt)
endif (${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_CURRENT_SOURCE_DIR})


# reset to zero when a project changes
INCLUDE_DIRECTORIES(
  ${MY_INCLUDE_DIRS}
  utilities
  graphics_utils
  cuda_utitilities
  random
  opencl_utilities
)

ADD_SUBDIRECTORY(utilities)
ADD_SUBDIRECTORY(graphics_utils)
#ADD_SUBDIRECTORY(cuda_utilities)
ADD_SUBDIRECTORY(random)
ADD_SUBDIRECTORY(opencl_utilities)

# examples require glew library, not installed on our systems.
#ADD_SUBDIRECTORY(examples)

#IF(PURE_OPENCL)
  #ADD_SUBDIRECTORY(examples_opencl)
#ENDIF(PURE_OPENCL)

#if (NOT ${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_CURRENT_SOURCE_DIR})
	#message("**** COMPILE swan_utilities ****")
	if (SWAN_CUDA OR SWAN_OPENCL)
		ADD_SUBDIRECTORY(swan_utilities)
	endif (SWAN_CUDA OR SWAN_OPENCL)
#endif (NOT ${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_CURRENT_SOURCE_DIR})

GET_DIRECTORY_PROPERTY(inc_examples_opencl DIRECTORY opencl_utilities INCLUDE_DIRECTORIES)
message("inc_examples_opencl(opencl_utilities): ${inc_examples_opencl}")
