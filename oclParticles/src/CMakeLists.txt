CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CXXFILES 
	paramgl.cpp
	particleSystemHost.cpp
	oclBitonicSort_launcher.cpp	
	particleSystem_class.cpp
	oclManager.cpp			
	render_particles.cpp
	oclParticles_launcher.cpp	
	shaders.cpp
	param.cpp
	timege.cpp
)


#ADD_DEFINITIONS(-DCL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/physics")
#ADD_DEFINITIONS(-DCL_SORT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/oclRadixSortKeysValues")

INCLUDE_DIRECTORIES(
	../inc
)

ADD_LIBRARY(particles ${CXXFILES})


