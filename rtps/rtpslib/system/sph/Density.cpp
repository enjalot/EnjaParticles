#include "../SPH.h"

namespace rtps {

Kernel SPH::loadEuler()
{
    #include "simple/euler.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_euler = Kernel(ps->cli, euler_program_source, "euler");
  
    //TODO: fix the way we are wrapping buffers
    k_euler.setArg(0, cl_position.cl_buffer[0]);
    k_euler.setArg(1, cl_velocity.cl_buffer[0]);
    k_euler.setArg(2, cl_force.cl_buffer[0]);
    k_euler.setArg(3, .01f); //time step (should be set from settings)

    return k_euler;
} 

}
