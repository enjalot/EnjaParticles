#include "../GE_SPH.h"
//#include "timer_macros.h"

namespace rtps {

void GE_SPH::computeCollisionWall()
{
	static bool first_time = true;

	ts_cl[TI_COLLISION_WALL]->start(); // OK

	if (first_time) {
		try {
			std::string path(CL_SPH_UTIL_SOURCE_DIR);
			printf("path= %s\n", path.c_str());
			path = path + "/collision_wall_cl.cl";
			printf("path= %s\n", path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	k_collision_wall = Kernel(ps->cli, strg, "collision_wall");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(computeEuler): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}
    //#include "collision_wall.cl"

	Kernel kern = k_collision_wall;
	int workSize = 128;
  
    //TODO: fix the way we are wrapping buffers
    //k_collision_wall.setArg(0, nb_vars);
    k_collision_wall.setArg(0, cl_vars_sorted->getDevicePtr());
    k_collision_wall.setArg(1, cl_GridParamsScaled->getDevicePtr());
    k_collision_wall.setArg(2, cl_params->getDevicePtr());

   	kern.execute(nb_el, workSize); 

    ps->cli->queue.finish();
	ts_cl[TI_COLLISION_WALL]->end(); // OK

} 

} // namespace
