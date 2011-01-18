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


    /*
    int nbc = 20;
    std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
    //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    for(int i = 0; i < nbc; i++)
    {
        printf("sh[%d] %d; ", i, sh[i]);
    }
    printf("\n");
    */


    //printf("about to data structures\n");
	int workSize = 64; // work group size
    try
    {
	    k_datastructures.execute(num, workSize);
    }
    catch (cl::Error er) {
        printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
	
    ps->cli->queue.finish();

#if 0 
    //printouts
    int nbc = 0;
    printf("start cell indices\n");
    printf("end cell indices\n");
    nbc = grid_params.nb_cells;
    std::vector<int> is = cl_cell_indices_start.copyToHost(nbc);
    std::vector<int> ie = cl_cell_indices_end.copyToHost(nbc);

    /*
    for(int i = 0; i < nbc; i++)
    {
        printf("sci[%d] %d eci[%d] %d\n", i, is[i], i, ie[i]);
    }
    */

    int nb_particles = 0;
    int nb;
    int asdf = 0;
    for (int i=0; i < grid_params.nb_cells; i++) {
    //for (int i=0; i < 100; i++) {
        //printf("is,ie[%d]= %d, %d\n", i, is[i], ie[i]);
        // ie[i] SHOULD NEVER BE ZERO 
        //printf("is[%d] %d ie[%d] %d\n", i, is[i], i, ie[i]);
        if (is[i] != -1 && ie[i] != 0) {
            nb = ie[i] - is[i];
            nb_particles += nb;
        }
        if (is[i] != -1 && ie[i] != 0 && i > 600 && i < 1000) { 
            asdf++;
            //printf("(GPU) [%d]: indices_start: %d, indices_end: %d, nb pts: %d\n", i, is[i], ie[i], nb);
        }
    }
    printf("asdf: %d\n", asdf);
    printf("done with data structures\n");
#endif

}



}
