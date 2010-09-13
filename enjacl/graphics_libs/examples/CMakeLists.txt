CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(graphics_library_examples)

ADD_DEFINITIONS(-g)

INCLUDE_DIRECTORIES(
	/usr/include
	${GLUT_INCLUDE_DIR}
	$ENV{CUDA_LOCAL}/common/inc
	#${CUDA_SDK_ROOT_DIR}/common/inc   # glew.h
	$ENV{GRAPHIC_LIBS_HOME}/graphics_utils
	$ENV{GRAPHIC_LIBS_HOME}/random
	$ENV{GRAPHIC_LIBS_HOME}/utilities
)

#ADD_SUBDIRECTORY(display_list)
ADD_SUBDIRECTORY(vbo)
ADD_SUBDIRECTORY(vbo_explicit)
ADD_SUBDIRECTORY(pingpong)

