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

	Kernel kern = k_neighbors;
	    	
    printf("setting kernel args\n");
	int iarg = 0;
	kern.setArg(iarg++, cl_vars_sorted.getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start.getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end.getDevicePtr());
	kern.setArg(iarg++, cl_GridParamsScaled.getDevicePtr());
	//kern.setArg(iarg++, cl_FluidParams->getDevicePtr());
	kern.setArg(iarg++, cl_SPHParams.getDevicePtr());

    /*
	// ONLY IF DEBUGGING
	kern.setArg(iarg++, clf_debug->getDevicePtr());
	kern.setArg(iarg++, cli_debug->getDevicePtr());
	//kern.setArg(iarg++, cl_index_neigh->getDevicePtr());
    */



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


	size_t global = (size_t) num;
	int local = 128;

 	k_neighbors.execute(global, local);
	ps->cli->queue.finish();
   
    /*
	if (which == 0) ts_cl[TI_DENS]->end();
	if (which == 1) ts_cl[TI_PRES]->end();
	if (which == 2) ts_cl[TI_COL]->end();
	if (which == 3) ts_cl[TI_COL_NORM]->end();
    */

#if 0
	if (which != 0) return;
	printf("============================================\n");
	printf("which == %d *** \n", which);

	clf_debug->copyToHost();
	cli_debug->copyToHost();
	float4* fclf = clf_debug->getHostPtr();
	int4*   icli = cli_debug->getHostPtr();

	cl_index_neigh->copyToHost();
	int* n = cl_index_neigh->getHostPtr();

	for (int i=0; i < nb_el; i++) { 
	//for (int i=0; i < 500; i++) { 
	//for (int i=500; i < 510; i++) { 
		printf("----------------------------\n");
		printf("clf[%d]= %f, %f, %f, %f\n", i, fclf[i].x, fclf[i].y, fclf[i].z, fclf[i].w);
		printf("cli[%d]= %d, %d, %d, %d\n", i, icli[i].x, icli[i].y, icli[i].z, icli[i].w);
		printf("index(%d): (%d)", i, icli[i].x); 
		int max = icli[i].x < 50 ? icli[i].x : 50;
		for (int j=0; j < icli[i].x; j++) {
			printf("%d, ", n[j+50*i]);
		}
		printf("\n");
	}
	//exit(0);
	#endif

}


} // namespace
