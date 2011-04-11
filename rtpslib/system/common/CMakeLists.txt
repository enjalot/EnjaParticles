CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FILE(GLOB cl_cpy "cl_src/*")

# copy the opencl source files to the build directory
MAKE_DIRECTORY(${rtpslib_BINARY_DIR}/cl_sph)

FOREACH(loop_var ${cl_cpy})
    FILE(COPY ${loop_var} DESTINATION ${rtpslib_BINARY_DIR}/cl_common)
ENDFOREACH(loop_var)

IF(RTPS_DEBUG)
# Only executes when I do "cmake"
# I'd like it to execute when I do a make
# this will run the c++ preprocessor on the opencl files
    IF(NOT WIN32)
        FILE(GLOB cl_src "cl_src/*.cl")
        #make_directory(${rtpslib_BINARY_DIR}/cl_sph)
        FOREACH(loop_var ${cl_src})
            get_filename_component(filename ${loop_var} NAME)
            #message("var : ${loop_var}")
            #message("cmd : ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x")
            EXECUTE_PROCESS(COMMAND ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x ${filename} ${rtpslib_BINARY_DIR}/cl_common
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cl_src
            )
            #message("working dir :  ${CMAKE_CURRENT_SOURCE_DIR}")
        ENDFOREACH(loop_var)
    ENDIF(NOT WIN32)
ENDIF(RTPS_DEBUG)

#IF(WIN32)
#install (TARGETS rtps DESTINATION bin)
install (FILES ${cl_cpy} DESTINATION ${rtpsuite_BINARY_DIR}/bin/cl_common)

#ENDIF(WIN32)
