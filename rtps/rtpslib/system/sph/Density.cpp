#include "../SPH.h"

namespace rtps {

void SPH::loadDensity()
{
    #include "density.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_density = Kernel(ps->cli, density_program_source, "density");
  
    //TODO: fix the way we are wrapping buffers
    k_density.setArg(0, cl_position.cl_buffer[0]);
    k_density.setArg(1, cl_density.cl_buffer[0]);
    k_density.setArg(2, cl_params.cl_buffer[0]);

} 

}
