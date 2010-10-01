#include "../GE_SPH.h"

namespace rtps {

void GE_SPH::loadCollision_wall()
{
    #include "collision_wall.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_collision_wall = Kernel(ps->cli, collision_wall_program_source, "collision_wall");
  
    //TODO: fix the way we are wrapping buffers
    k_collision_wall.setArg(0, cl_position->getDevicePtr());
    k_collision_wall.setArg(1, cl_velocity->getDevicePtr());
    k_collision_wall.setArg(2, cl_force->getDevicePtr());
    k_collision_wall.setArg(3, cl_params->getDevicePtr());

} 

}
