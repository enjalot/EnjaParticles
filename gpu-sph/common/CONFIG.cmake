

# define variables for: 
#   graphics_libs includes and libraries 
#   graphics libraries

# SOME HARD-CODED PATHS NECESSARY:
# assumes that graphics libs is a separate project
SET(GLIB ${gravimetrie_SOURCE_DIR}/graphics_libs)
SET(GLIB_BIN ${gravimetrie_BINARY_DIR}/graphics_libs)
ADD_DEFINITIONS(-g)


#---
SET(GRAPHICS_LIBS_INCLUDES 
   ${GLIB}/graphics_utils
   ${GLIB}/random
   ${GLIB}/utilities
   ${GLIB}/include
)

message("******** GRAPHICS_LIBS_INCLUDES= ${GRAPHICS_LIBS_INCLUDES}")

SET(GRAPHICS_LIBS_LIBRARIES
  ${GLIB_BIN}/graphics_utils/libgraphic_utilities.a
  ${GLIB_BIN}/random/librandom.a
  ${GLIB_BIN}/utilities/libutilities.a
)

message("+++***TOP, INCLUDES= ${GRAPHICS_LIBS_INCLUDES} ***+++")

SET(SWAN_INCLUDES ${SWAN}/include)

SET(OPENCL_INCLUDES ${OPENCL_INCLUDE_DIR})
# I do not believe there are OpenCL libraries

set (MY_CUDA_LIBRARIES cuda cudart ${CUDA_CUT_LIBRARIES})
#set (MY_CUDA_LIBRARIES cublas cuda cudart ${CUDA_CUT_LIBRARIES})
set (MY_CUDA_INCLUDES ${CUDA_INCLUDE_DIRS} $ENV{CUDA_LOCAL}/common/inc)
find_path(glew_inc_path GL/glew.h ${MY_CUDA_INCLUDES} )


SET(PLIB $ENV{CUDA_LOCAL}/common/lib)
FIND_LIBRARY(cudpp cudpp_i386 PATHS ${PLIB}/darwin ${PLIB}/linux)
FIND_LIBRARY(cutil cutil_i386 PATHS $ENV{CUDA_LOCAL}/lib)

#---
IF (PURE_CUDA)
  message("+++***PURE_CUDA, INCLUDES= ${GRAPHICS_LIBS_INCLUDES} ***+++")
  ADD_DEFINITIONS(-DCUDA)
  SET(GRAPHICS_LIBS_INCLUDES ${GRAPHICS_LIBS_INCLUDES} ${GLIB}/cuda_utilities)
  SET(GRAPHICS_LIBS_LIBRARIES 
  	${GRAPHICS_LIBS_LIBRARIES} 
	${GLIB_BIN}/cuda_utilities/libcuda_utilities_cpp.a
	${GLIB_BIN}/cuda_utilities/libcuda_utilities_cu.a
  )
  SET(MY_INCLUDE_DIRS 
    ${GRAPHICS_LIBS_INCLUDES}
	${CUDA_SDK_ROOT_DIR}/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/include
  )
  SET(MY_LIBRARIES 
  	${GRAPHICS_LIBS_LIBRARIES} 
	${MY_CUDA_LIBRARIES}
  )
ENDIF (PURE_CUDA)

#---
IF (PURE_OPENCL)
  message("+++***PURE_OPENCL, INCLUDES= ${GRAPHICS_LIBS_INCLUDES} ***+++")
  ADD_DEFINITIONS(-DOPENCL)
  SET(GRAPHICS_LIBS_INCLUDES ${GRAPHICS_LIBS_INCLUDES} ${GLIB}/opencl_utilities)
  SET(GRAPHICS_LIBS_LIBRARIES 
  	${GRAPHICS_LIBS_LIBRARIES} 
	${GLIB_BIN}/opencl_utilities/libcl_utilities_cpp.a
  )
  SET(MY_INCLUDE_DIRS ${GRAPHICS_LIBS_INCLUDES} ${OPENCL_INCLUDE_DIR})
  SET(MY_LIBRARIES 
  	${GRAPHICS_LIBS_LIBRARIES} 
	${MY_CUDA_LIBRARIES}
  )
ENDIF (PURE_OPENCL)

#---

