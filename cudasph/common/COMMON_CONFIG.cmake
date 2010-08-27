###############################################
# Build Options (Definitions and compiler flags)
###############################################
	# Used by ALL compilers
	#ADD_DEFINITIONS(-g)
	# Used by SPECIFIC compilers
	# SET (CMAKE_CXX_FLAGS )


###############################################
# EXTENSIONS TO INCLUDE: 
###############################################
	ENABLE_TESTING()
	#INCLUDE (CPack)
	#INCLUDE (UseDoxygen)
	#FIND_PACKAGE (MPI)

	FIND_PACKAGE (GLUT)
	FIND_PACKAGE (OpenGL)
	FIND_PACKAGE (OPENCL)

###############################################
# External dependency search paths
###############################################
	# Directories searched for headers ORDER does not matter. 
	# If a directory does not exist it is skipped
	#set ( GLIB $ENV{GRAPHIC_LIBS_HOME} )

###############################################
# Locate Required Libraries
###############################################
	# Find library: find_library(<VAR> name1 [path1 path2 ...])
    
# CHECK on Linux systems. 
IF(APPLE)
	SET(stdc /usr/local/lib/i386/libstdc++.a)
ELSE(APPLE)
	#FIND_LIBRARY(stdc stdc++ PATH /usr/lib32) #/gcc/x86_64-linux-gnu/4.4/32/)
	SET(stdc /usr/lib/gcc/x86_64-linux-gnu/4.4/32/libstdc++.a)
message("************* stdc (non-apple)= ${stdc} ************")
ENDIF(APPLE)


#LINK_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/lib ${LINK_DIRECTORIES} )

# might not be found. Should be compiled ahead of time
#if (PURE_OPENCL) 
	#FIND_LIBRARY(ocl_common ocl_common PATH ${CMAKE_CURRENT_SOURCE_DIR}/graphics_lib/examples_opencl/cublas/)
#endif (PURE_OPENCL) 

