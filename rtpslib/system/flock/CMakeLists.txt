CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FILE(GLOB cl_cpy "cl_src/*")

# copy the opencl source files to the build directory
MAKE_DIRECTORY(${rtpslib_BINARY_DIR}/cl_flock)

FOREACH(loop_var ${cl_cpy})
    FILE(COPY ${loop_var} DESTINATION ${rtpslib_BINARY_DIR}/cl_flock)
ENDFOREACH(loop_var)

IF(PREPROCESS_CL)
# Only executes when I do "cmake"
# I'd like it to execute when I do a make
# this will run the c++ preprocessor on the opencl files
    IF(NOT WIN32)
        FILE(GLOB cl_src "${rtpslib_BINARY_DIR}/cl_flock/*.cl")
        FILE(MAKE_DIRECTORY ${rtpslib_BINARY_DIR}/cl_flock/tmp)
        #make_directory(${rtpslib_BINARY_DIR}/cl_flock)
        FOREACH(loop_var ${cl_src})
            get_filename_component(filename ${loop_var} NAME)
            #message("filename:  ${filename}")
            #message("dir: ${rtpslib_BINARY_DIR}/cl_flock")
            #message("var : ${loop_var}")
            #message("cmd : ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x")
            EXECUTE_PROCESS(COMMAND ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x ${filename} ${rtpslib_BINARY_DIR}/cl_flock/tmp
                WORKING_DIRECTORY ${rtpslib_BINARY_DIR}/cl_flock
                #WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cl_src
            )
            #message("working dir :  ${CMAKE_CURRENT_SOURCE_DIR}")
        ENDFOREACH(loop_var)
        
        FILE(GLOB cl_cpy "${rtpslib_BINARY_DIR}/cl_flock/tmp/*")
        FOREACH(loop_var ${cl_cpy})
            FILE(COPY ${loop_var} DESTINATION ${rtpslib_BINARY_DIR}/cl_flock)
        ENDFOREACH(loop_var)

        FILE(REMOVE_RECURSE ${rtpslib_BINARY_DIR}/cl_flock/tmp)


    ENDIF(NOT WIN32)
ENDIF(PREPROCESS_CL)

#IF(WIN32)
#install (TARGETS rtps DESTINATION bin)
INSTALL(CODE "FILE(MAKE_DIRECTORY ${RTPS_INSTALL_DIR}/cl_flock)")
FILE(GLOB cl_cpy "${rtpslib_BINARY_DIR}/cl_flock/*")
install (FILES ${cl_cpy} DESTINATION ${RTPS_INSTALL_DIR}/cl_flock)

#ENDIF(WIN32)
