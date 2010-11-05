#include "SPH.h"

#include <string>

namespace rtps {

void SPH::loadPrep()
{
    printf("create prep kernel\n");
    std::string path(SPH_CL_SOURCE_DIR);
	path = path + "/prep_cl.cl";
    k_prep = Kernel(ps->cli, path, "prep");

    int args = 0;
	k_prep.setArg(args++, cl_density.getDevicePtr());
	k_prep.setArg(args++, cl_position.getDevicePtr());
	k_prep.setArg(args++, cl_velocity.getDevicePtr());
	k_prep.setArg(args++, cl_veleval.getDevicePtr());
    k_prep.setArg(args++, cl_vars_unsorted.getDevicePtr()); // positions + other variables
	//k_prep.setArg(args++, cl_sort_indices.getDevicePtr());


}

void SPH::prep()
{

	int ctaSize = 128; // work group size
	// Hash based on unscaled data
	k_prep.execute(num, ctaSize);
	
    ps->cli->queue.finish();
}

}
