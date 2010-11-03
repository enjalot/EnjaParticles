CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# List of files to precompile
SET(FILES
   euler_cl
)  # ignore cpp extension

# Only executes when I do "cmake"
# I'd like it to execute when I do a make


FOREACH(loop_var ${FILES})
    #message("var : ${loop_var}")
    #message("cmd : ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x")
    EXECUTE_PROCESS(COMMAND ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x ${loop_var}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    #message("working dir :  ${CMAKE_CURRENT_SOURCE_DIR}"
ENDFOREACH(loop_var)

