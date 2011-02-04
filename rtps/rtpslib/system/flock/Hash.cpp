#include "FLOCK.h"

#include <string>

namespace rtps {

void FLOCK::loadHash()
{
    printf("create hash kernel\n");
    std::string path(FLOCK_CL_SOURCE_DIR);
	path = path + "/hash_cl.cl";
    k_hash = Kernel(ps->cli, path, "hash");

    printf("kernel made, set args\n");
    int args = 0;
    k_hash.setArg(args++, num); 
    k_hash.setArg(args++, cl_vars_unsorted.getDevicePtr()); // positions + other variables
	k_hash.setArg(args++, cl_sort_hashes.getDevicePtr());
	k_hash.setArg(args++, cl_sort_indices.getDevicePtr());
	k_hash.setArg(args++, cl_GridParams.getDevicePtr());
    k_hash.setArg(args++, clf_debug.getDevicePtr());
	k_hash.setArg(args++, cli_debug.getDevicePtr());


}

void FLOCK::hash()
// Generate hash list: stored in cl_sort_hashes
{

	int ctaSize = 128; // work group size
	// Hash based on unscaled data
    k_hash.setArg(0, num); 
	k_hash.execute(num, ctaSize);
	// set cell_indicies_start to -1
	int minus = 0xffffffff;

	ps->cli->queue.finish();

	//-------------------
	// Set cl_cell indices to -1
    std::vector<int> cells_indices_start(grid_params.nb_cells);
    std::fill(cells_indices_start.begin(), cells_indices_start.end(), minus);
	cl_cell_indices_start.copyToDevice(cells_indices_start);

	//-------------------

	//sset(gp->nb_points, minus, cl_cell_indices_start->getDevicePtr());
	//exit(0);   // SOMETHING WRONG with sset!!! WHY? 

    //printHashDiagnostics();

	#if 0
	GridParams& gp = *cl_GridParams->getHostPtr();
	cl_sort_hashes->copyToHost();
	int* h = cl_sort_hashes->getHostPtr();
	int sz = (int) (gp.grid_res.x * gp.grid_res.y * gp.grid_res.z);
	int mx = -1;
	for (int i=0; i < nb_el; i++) {
		//printf("h[%d]= %d\n", i, h[i]);
		if (h[i] > mx) mx = h[i];
	}
	printf("sz= %d, max hash: %d\n", sz, mx);
	//exit(0);
	#endif
}

//----------------------------------------------------------------------

void FLOCK::printHashDiagnostics()
{
#if 1
	printf("***** PRINT hash diagnostics ******\n");
    std::vector<int> sh = cl_sort_hashes.copyToHost(num);
    std::vector<int> si = cl_sort_indices.copyToHost(num);
	//cl_cells->copyToHost();
    std::vector<int4> cli = cli_debug.copyToHost(num);
    std::vector<float4> clf = clf_debug.copyToHost(num);
	//cl_GridParams.copyToHost();

	//GridParams& gp = *cl_GridParams->getHostPtr();
	//gp.print();

	//cli_debug->copyToHost();

	//for (int i=0; i < num; i++) {  
	for (int i=0; i < 10; i++) {  
		printf(" cl_sort_hash[%d] %u, cl_sort_indices[%d]: %u\n", i, sh[i], i, si[i]);
		printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
		printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
		printf("-----\n");

		#if 0
		int gx = (cl_cells[i].x - gp.grid_min.x) * gp.grid_inv_delta.x ;
		int gy = (cl_cells[i].y - gp.grid_min.y) * gp.grid_inv_delta.y ;
		int gz = (cl_cells[i].z - gp.grid_min.z) * gp.grid_inv_delta.z ;
		//printf("cl_cells,cl_cells,cl_cells= %f, %f, %f\n", cl_cells[i].x, cl_cells[i].y, cl_cells[i].z);
		//gp.grid_min.print("grid min");
		//printf("gx,gy,gz= %d, %d, %d\n", gx, gy, gz);
		unsigned int idx = (gz*gp.grid_res.y + gy) * gp.grid_res.x + gx; 
		if (idx != cl_sort_hashes[i]) {
			printf("hash indices (exact vs GPU do not match)\n");
		}
		printf("cli_debug: %d, %d, %d\n", cli_debug[i].x, cli_debug[i].y, cli_debug[i].z);
		//printf("---------------------------\n");
		#endif
	}
#endif
}

//----------------------------------------------------------------------

}
