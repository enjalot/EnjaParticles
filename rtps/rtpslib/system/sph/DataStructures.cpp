#include "SPH.h"

#include <string>

namespace rtps {

void SPH::loadDataStructures()
{
    printf("create datastructures kernel\n");
    std::string path(SPH_CL_SOURCE_DIR);
	path = path + "/datastructures_cl.cl";
    k_datastructures = Kernel(ps->cli, path, "datastructures");

    int iarg = 0;
    k_datastructures.setArg(iarg++, cl_vars_unsorted.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_vars_sorted.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_sort_hashes.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_sort_indices.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_cell_indices_start.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_cell_indices_end.getDevicePtr());
	k_datastructures.setArg(iarg++, cl_SPHParams.getDevicePtr());
	//k_datastructures.setArg(iarg++, cl_GridParamsScaled->getDevicePtr());
    
    int workSize = 64;
	int nb_bytes = (workSize+1)*sizeof(int);
    k_datastructures.setArgShared(iarg++, nb_bytes);

}

void SPH::buildDataStructures()
// Generate hash list: stored in cl_sort_hashes
{

	int workSize = 64; // work group size
	k_datastructures.execute(num, workSize);

	ps->cli->queue.finish();

}

}
