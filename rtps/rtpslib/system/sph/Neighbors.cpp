#include "SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------

//----------------------------------------------------------------------
void SPH::loadNeighbors()
{
	printf("enter neighbor\n");

    try {
        string path(SPH_CL_SOURCE_DIR);
        path = path + "/neighbors_cl.cl";
        k_neighbors = Kernel(ps->cli, path, "neighbors");
        printf("bigger problem\n");
    } catch(cl::Error er) {
        printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
        exit(1);
    }
    printf("ASDFSDFSF\n");

	Kernel kern = k_neighbors;
	    	
    printf("setting kernel args\n");
	int iarg = 0;
	kern.setArg(iarg++, cl_vars_sorted.getDevicePtr());
	kern.setArg(iarg++, cl_ghosts_sorted.getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start.getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end.getDevicePtr());
	kern.setArg(iarg++, cl_GridParamsScaled.getDevicePtr());
	//kern.setArg(iarg++, cl_FluidParams->getDevicePtr());
	kern.setArg(iarg++, cl_SPHParams.getDevicePtr());

	// ONLY IF DEBUGGING
	kern.setArg(iarg++, clf_debug.getDevicePtr());
	kern.setArg(iarg++, cli_debug.getDevicePtr());
	//kern.setArg(iarg++, cl_index_neigh->getDevicePtr());



	}
//----------------------------------------------------------------------

void SPH::neighborSearch(int choice)
{

	// which == 0 : density update
	// which == 1 : force update

    /*
	if (which == 0) ts_cl[TI_DENS]->start();
	if (which == 1) ts_cl[TI_PRES]->start();
	if (which == 2) ts_cl[TI_COL]->start();
	if (which == 3) ts_cl[TI_COL_NORM]->start();
    */

    //Copy choice to SPHParams
	params.choice = choice;
    std::vector<SPHParams> vparams(0);
    vparams.push_back(params);
    cl_SPHParams.copyToDevice(vparams);

    /*
    std::vector<int4> cli = cli_debug.copyToHost(2);
    for (int i=0; i < 2; i++) 
    {  
		printf("cli_debug: %d\n", cli[i].w);
    }
    */

	size_t global = (size_t) num;
	int local = 128;

    try{
 	k_neighbors.execute(num, local);
    }

    catch (cl::Error er) {
        printf("ERROR(neighbor %d): %s(%s)\n", choice, er.what(), oclErrorString(er.err()));
    }
	ps->cli->queue.finish();
#if 0
    if(choice == 1)
    {
        std::vector<int4> cli = cli_debug.copyToHost(10);
        std::vector<float4> clf = clf_debug.copyToHost(10);
        for (int i=0; i < 1; i++) 
        {  
            printf("num force calcs: %d\n", cli[10].w);
		    printf("cli_debug: %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z);
		    printf("clf_debug: %f, %f, %f %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
        }
    }
#endif

#if 0 //printouts    
    //DEBUGING
	printf("============================================\n");
	printf("which == %d *** \n", choice);

	printf("***** PRINT neighbors diagnostics ******\n");
    cli = cli_debug.copyToHost(num);
    std::vector<float4> clf = clf_debug.copyToHost(num);

	//for (int i=0; i < num; i++) {  
	for (int i=0; i < 10; i++) 
    {  
		printf("-----\n");
		printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
		printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
    }
#endif

}


} // namespace
