#include "../SPH.h"

namespace rtps {

void SPH::loadPressure()
{
    #include "pressure.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_pressure = Kernel(ps->cli, pressure_program_source, "pressure");
  
    //TODO: fix the way we are wrapping buffers
    k_pressure.setArg(0, cl_position.cl_buffer[0]);
    k_pressure.setArg(1, cl_density.cl_buffer[0]);
    k_pressure.setArg(2, cl_force.cl_buffer[0]);
    k_pressure.setArg(3, cl_params.cl_buffer[0]);

} 

}
