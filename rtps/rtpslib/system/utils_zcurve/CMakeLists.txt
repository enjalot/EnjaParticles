CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(enjacl)


# List of files to precompile
SET(FILES
   uniform_grid_utils
   neighbors
   uniform_hash
   datastructures_test
   euler_cl
   leapfrog_cl
   collision_wall_cl
   block_scan_cl
)  # ignore cpp extension

# Only executes when I do "cmake"
# I'd like it to execute when I do a make

FOREACH(loop_var ${FILES})
    message("var : ${loop_var}")
    EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile_to_cl.x ${loop_var}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDFOREACH(loop_var)

