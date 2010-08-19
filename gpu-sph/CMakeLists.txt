
# C is not required on the mac, but appears to be on linux
project(blenderparticles C CXX)

IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE(FATAL_ERROR "CMake generation must always be out-of-source!
    Remove the CMakeCache.txt file and try again from another folder")
ENDIF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


# set to 64 bit
SET(CUDA_64_BIT_DEVICE_CODE OFF)

# options controlled via -D option to cmake
# Only a single option is on at a given time. 
# All libraries (including graphics_libs) are recreated for each case to
#   decrease the chance of mishap
# DANGER: If one of these is ON, and I use a script that turns another one 
# ON, there might be inconsistencies in the Makefile generation. 
OPTION(PURE_CUDA   "Pure Cuda"   ON)


IF (PURE_CUDA)
  #ADD_DEFINITIONS(-DDOUBLE)
ENDIF (PURE_CUDA)


################ BEGIN OS DEPENDENT CONFIGS ###################

SET( CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
	$ENV{CMAKE_MODULES}
    #${CMAKE_CURRENT_SOURCE_DIR}/common/UseDoxygen
    #${CMAKE_CURRENT_SOURCE_DIR}/common/trilinos_cmake
)

###############################################
#       APPLE OSX 10.6
###############################################
IF(APPLE)
	message("apple")
    INCLUDE (OSX_CONFIG)

ELSEIF($ENV{HOSTNAME} MATCHES "mark2")
	message("mark2")
	INCLUDE(MARK2_CONFIG)

###############################################
#       UBUNTU LINUX 9.10
#       FSU Vislab
###############################################
ELSEIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	message("ubuntu")
    INCLUDE (UBUNTU_CONFIG)

ELSE(APPLE)
	message(FATAL_ERROR "Not implemented")

ENDIF(APPLE)

# **** put elsewhere! ***
# double precision kernel

################# END OS DEPENDENT CONFIGS ####################


# Get common build instructions used by all operating systems
INCLUDE (COMMON_CONFIG)
INCLUDE (CONFIG)

# Valid for all projects 
# missing: specialized gpu libraries. 
INCLUDE_DIRECTORIES(
	${MY_INCLUDE_DIRS}
)

# Projectd gravimetrie_fortran
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/SPHSimLib/)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/ConsoleTest/)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/enj_standalone/)

