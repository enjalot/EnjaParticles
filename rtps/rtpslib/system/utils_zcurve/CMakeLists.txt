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
   block_scan_block64_cl
   block_scan_block64a_cl
   block_scan_loop_cl
   block_scan_pres_cl
   sum_cl
   compactify_cl
   compactify_down_cl
   compactify_middle_cl
   compactify_sub1_cl
   compactify_sub2_cl
   compactify_sub2sum_cl
   compactify_sub3_cl
   compactify_sub4_cl
   compactify_sub4int4_cl
   sum_scan_single_block_cl
)  # ignore cpp extension

# Only executes when I do "cmake"
# I'd like it to execute when I do a make

FOREACH(loop_var ${FILES})
    message("var : ${loop_var}")
    EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile_to_cl.x ${loop_var}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDFOREACH(loop_var)

