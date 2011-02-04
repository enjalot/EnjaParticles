CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# List of files to precompile
SET(FILES
    density_cl
    pressure_cl
    viscosity_cl
    xflock_cl
    leapfrog_cl
    euler_cl
    collision_wall_cl
    prep_cl
    hash_cl
    datastructures_cl
    neighbors_cl
    scopy_cl
)  # ignore cpp extension

# Only executes when I do "cmake"
# I'd like it to execute when I do a make

make_directory(${rtpslib_BINARY_DIR}/cl_flock)
FOREACH(loop_var ${FILES})
    #message("var : ${loop_var}")
    #message("cmd : ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x")
    EXECUTE_PROCESS(COMMAND ${rtpslib_SOURCE_DIR}/utils/compile_to_cl.x ${loop_var} ${rtpslib_BINARY_DIR}/cl_flock
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    #message("working dir :  ${CMAKE_CURRENT_SOURCE_DIR}")
ENDFOREACH(loop_var)

