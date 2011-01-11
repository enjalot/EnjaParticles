#####################################
#this file sets
#MY_CUDA_INCLUDES
#and
#MY_CUDA_LIBRARIES
#Ian Johnson and Gordon Erlebacher
#####################################



set (CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_INSTALL_PATH})  # set environment variable
set (CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_INSTALL_PATH})  # set environment variable

SET(PLIB ${CUDA_SDK_ROOT_DIR}/common/lib)
message("PLIB: ${PLIB}\n")

#SET(CUDA_64_BIT_DEVICE_CODE ON)
IF(CUDA_64_BIT_DEVICE_CODE)
    FIND_PACKAGE(CUDA_64)
    message("FOUND CUDA 64\n")
    FIND_LIBRARY(cudpp cudpp_x86_64 PATHS ${PLIB}/darwin ${PLIB}/linux)
    FIND_LIBRARY(cutil cutil_x86_64 PATHS ${CUDA_SDK_ROOT_DIR}/lib)
    #set(cuda_cutil_name cutil_x86_64)
ELSE(CUDA_64_BIT_DEVICE_CODE)
    FIND_PACKAGE(CUDA)
    FIND_LIBRARY(cudpp cudpp_i386 PATHS ${PLIB}/darwin ${PLIB}/linux)
    FIND_LIBRARY(cutil cutil_i386 PATHS ${CUDA_SDK_ROOT_DIR}/lib)
    #set(cuda_cutil_name cutil_i386)
ENDIF(CUDA_64_BIT_DEVICE_CODE)




message("CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")

find_library(CUDA_CUT_LIBRARY
   NAMES cutil ${cuda_cutil_name}
   PATHS ${CUDA_SDK_ROOT_DIR}
   # The new version of the sdk shows up in common/lib, but the old one is in lib
   PATH_SUFFIXES "common/lib" "lib"
   DOC "Location of cutil library"
   NO_DEFAULT_PATH
 )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})

MESSAGE("+++++++++++++++++++++++++++++++++")
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

set (MY_CUDA_LIBRARIES cuda ${cudpp} ${cutil} ${CUDA_CUDART_LIBRARY}) #${CUDA_CUT_LIBRARIES})
#set (MY_CUDA_LIBRARIES cublas cuda cudart ${CUDA_CUT_LIBRARIES})
#set (MY_CUDA_INCLUDES ${CUDA_INCLUDE_DIRS} $ENV{CUDA_LOCAL}/common/inc)
SET(MY_CUDA_INCLUDES
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_SDK_ROOT_DIR}/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

ADD_DEFINITIONS(-DCUDA)

################ BEGIN OS DEPENDENT CONFIGS ###################
    ###############################################
    #       APPLE OSX 10.6
    ###############################################
    IF(APPLE)
        message("apple")
        set(CUDA_NVCC_FLAGS -Xcompiler -m32)

    ###############################################
    #       UBUNTU LINUX 9.10
    #       FSU Vislab
    ###############################################
    ELSEIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        message("ubuntu")
        set(CUDA_NVCC_FLAGS -Xcompiler -D__builtin_stdarg_start=__builtin_va_start)

    ELSE(APPLE)
        message(FATAL_ERROR "Not implemented")

    ENDIF(APPLE)

################# END OS DEPENDENT CONFIGS ####################
