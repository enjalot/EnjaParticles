
cmake_minimum_required(VERSION 2.6)
project(amira)
set(amira WriteAmiraMesh.cpp ReadRectilinearAmiraMesh.cpp write_lineset.cpp Endian.cpp
   WriteSurfaceAmiraMesh.cpp WriteHyperSurface.cpp write_psi.cpp )

ADD_DEFINITIONS(-DGORDON_FOURBYTEINT)
ADD_DEFINITIONS(-g)
ADD_DEFINITIONS(-arch i386)

include_directories( . ../utilities
	/Users/erlebach/Documents/src/graphics_libs/ )   # for my mac
add_library(amira STATIC ${amira})

