CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# List of files to precompile

SET(FILES
    density.cl
    pressure.cl
    viscosity.cl
    xsph.cl
    leapfrog.cl
    euler.cl
    collision_wall.cl
    collision_tri.cl
    prep.cl
    hash.cl
    datastructures.cl
    neighbors.cl
    scopy.cl
)  # ignore cpp extension

FILE(GLOB cl_src "cl_src/*.cl")
#FILE(GLOB cl_header "cl_src/*.h")
FILE(GLOB cl_cpy "cl_src/*")

# copy the opencl source files to the build directory
MAKE_DIRECTORY(${rtpslib_BINARY_DIR}/cl_sph)

FOREACH(loop_var ${cl_cpy})
    FILE(COPY ${loop_var} DESTINATION ${rtpslib_BINARY_DIR}/cl_sph)
ENDFOREACH(loop_var)

IF(RTPS_DEBUG)
# Only executes when I do "cmake"
# I'd like it to execute when I do a make
# this will run the c++ preprocessor on the opencl files
    IF(NOT WIN32)
        #make_directory(${rtpslib_BINARY_DIR}/cl_sph)
        FOREACH(loop_var ${cl_src})
            #message("var : ${loop_var}")
            #message("cmd : ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x")
            EXECUTE_PROCESS(COMMAND ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x ${loop_var} ${rtpslib_BINARY_DIR}/cl_sph
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cl_src
            )
            #message("working dir :  ${CMAKE_CURRENT_SOURCE_DIR}")
        ENDFOREACH(loop_var)
    ENDIF(NOT WIN32)
ENDIF(RTPS_DEBUG)
