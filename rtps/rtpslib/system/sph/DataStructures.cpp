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


    int nbc = 20;
    std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
    //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    for(int i = 0; i < nbc; i++)
    {
        printf("sh[%d] %d\n", i, sh[i]);
    }



    printf("about to data structures\n");
	int workSize = 64; // work group size
    try
    {
	    k_datastructures.execute(num, workSize);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
	
    printf("start cell indices\n");
    printf("end cell indices\n");
    //int nbc = 20;
    std::vector<int> sci = cl_cell_indices_start.copyToHost(nbc);
    std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    for(int i = 0; i < nbc; i++)
    {
        printf("sci[%d] %d eci[%d] %d\n", i, sci[i], i, eci[i]);
    }



    ps->cli->queue.finish();

    printf("done with data structures\n");


}

}
