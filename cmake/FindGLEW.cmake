#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
# 

IF (WIN32)
message(GLEW path "$ENV{GLEW_PATH}")
    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
        $ENV{PROGRAMFILES}/GLEW/include
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
	    ~/glew-1.5.8/include
	    #$ENV{GLEW_PATH}/include
	    $ENV{NVSDKCOMPUTE_ROOT}/shared/inc
        DOC "The directory where GL/glew.h resides")

    FIND_LIBRARY( GLEW_LIBRARY
        NAMES glew GLEW glew32 glew32s 
        PATHS
	    ~/glew-1.5.8/bin
	    ~/glew-1.5.8/lib
        $ENV{PROGRAMFILES}/GLEW/lib
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
	    #$ENV{GLEW_PATH}/lib
	    $ENV{NVSDKCOMPUTE_ROOT}/shared/lib
        DOC "The GLEW library")

ELSE (WIN32)
    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARY
        NAMES GLEW glew
        PATHS
		# path on Gordon's notebook mac
		/opt/local/lib  # 64bit on mac notebook
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        DOC "The GLEW library"
		NO_DEFAULT_PATH)
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH)
    SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
    SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )
