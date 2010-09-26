#include "../SPH.h"

namespace rtps {

void SPH::loadViscosity()
{
    #include "viscosity.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_viscosity = Kernel(ps->cli, viscosity_program_source, "viscosity");
  
    //TODO: fix the way we are wrapping buffers
    k_viscosity.setArg(0, cl_position.cl_buffer[0]);
    k_viscosity.setArg(1, cl_velocity.cl_buffer[0]);
    k_viscosity.setArg(2, cl_density.cl_buffer[0]);
    k_viscosity.setArg(3, cl_force.cl_buffer[0]);
    k_viscosity.setArg(4, cl_params.cl_buffer[0]);

} 

}
