cmake_minimum_required(VERSION 2.8)

ADD_SUBDIRECTORY(utilities)
SET(SRC main.cpp structs.cpp domain/IV.cpp boids.cpp)
INCLUDE_DIRECTORIES(. domain ian utilities)


FIND_PACKAGE (OpenGL)
FIND_PACKAGE (GLUT)
#FIND_PACKAGE (OPENCL)
#FIND_PACKAGE (GLEW)
#FIND_PACKAGE (GLU)

#SET(GUTILS $ENV{GRAPHICS_LIBS}/graphics_utils)
#SET(GRAND $ENV{GRAPHICS_LIBS}/random)

#ADD_LIBRARY(LB
	#main
	#domain/IV
#)

ADD_EXECUTABLE(eexec ${SRC})

TARGET_LINK_LIBRARIES( eexec 
   	${GLUT_LIBRARIES}
   	${OPENGL_LIBRARIES}
   	#${OPENCL_LIBRARIES}
   	#${GLEW_LIBRARY}
	#${GLU_LIBRARY}
	${LB}
    ${utilities}
)

IF (USE_SWIG) 
    ADD_SUBDIRECTORY (swig)
ENDIF (USE_SWIG)
