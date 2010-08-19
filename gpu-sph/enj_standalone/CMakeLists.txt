
MESSAGE("cuda: ${CUDA_64_BIT_DEVICE_CODE}")

SET(CXXFILES 
	EnjaCudaHelper.cpp
	EnjaSimBuffer.cpp
)

INCLUDE_DIRECTORIES(
	  $ENV{CUDA_LOCAL}/../shared/inc
	  ${MY_CUDA_INCLUDES}
      ${OPENGL_INCLUDE_DIR}
      ${CUDA_SDK_ROOT_DIR}/common/inc
	  ${CUDA_TOOLKIT_ROOT_DIR}/include
	  ${CMAKE_CURRENT_SOURCE_DIR}/../SPHSimLib
	  ${CMAKE_CURRENT_SOURCE_DIR}/../../enjacl
	  ${CMAKE_CURRENT_SOURCE_DIR}/../../opencl10
)

ADD_LIBRARY(enja ${CXXFILES})


LINK_DIRECTORIES(
	 ${CMAKE_CURRENT_SOURCE_DIR}/../../enjacl
     $ENV{CUDA_LOCAL}/lib 
	 $ENV{CUDA_LOCAL}/common/lib 
	 $ENV{CUDA_LOCAL}/../shared/lib/darwin 
	 $ENV{CUDA_LOCAL}/../shared/lib/linux
	 $ENV{CUDA_HOME}/lib
)


ADD_EXECUTABLE(sph_enja 
   main.cpp
)

FIND_PACKAGE(GLEW)

message("... ${GLEW_LIBRARY} ...")
message("... ${GLUT_LIBRARIES} ...")
message("... ${OPENGL_LIBRARIES} ...")

TARGET_LINK_LIBRARIES(sph_enja
	enja
	enjaclparticles     # make sure library is in source directory
	sph_cuda
	sph_lib
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${OPENCL_LIBRARIES}
   ${MY_CUDA_LIBRARIES}
   ${GLEW_LIBRARY}
)

