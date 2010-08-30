SET (CPPFILES 
	abstractintegrator.cpp
	cldevice.cpp
	cllink.cpp
	clplatform.cpp
	clprogram.cpp
	clsubprogram.cpp
	clsystem.cpp
	clvariable.cpp
	cpubitonicsort.cpp
	diffusionsimulation.cpp
	geometry.cpp
	gpubitonicsort.cpp
	log.cpp
	particle.cpp
	#pcintegrator.cpp
	pcisphsimulation.cpp
	probemanager.cpp
	#simulation.cpp
	stdwriter.cpp
	utils.cpp
	vec.cpp
	vtkwriter.cpp
	#wcsphsimulation.cpp
	writer.cpp
)

INCLUDE_DIRECTORIES(
	.
)

LINK_DIRECTORIES(
)

ADD_LIBRARY(sph ${CPPFILES})
message("*** bin dir: ${CMAKE_CURRENT_BINARY_DIR}")
#FIND_LIBRARY(SPH sph PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(SPH ${CMAKE_CURRENT_BINARY_DIR}/libsph.a)
message("*** SPH: ${SPH}")

