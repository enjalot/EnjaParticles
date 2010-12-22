#include "SPH.h"

#include <string>

namespace rtps {


void SPH::sortGhosts()
{

    cl_ghosts.acquire();
    printf("ghost hash\n");
    ghost_hash();
    printf("bitonic sort\n");
    bitonic_sort();
    printf("ghost data structures\n");
    build_ghost_datastructures();
    printf("ghost release\n");

    cl_ghosts.release();
}

void SPH::loadGhostHash()
{
    printf("create ghost hash kernel\n");
    std::string path(SPH_CL_SOURCE_DIR);
	path = path + "/ghost_hash_cl.cl";
    k_ghost_hash = Kernel(ps->cli, path, "ghost_hash");

    printf("kernel made, set args\n");
    int args = 0;
    k_ghost_hash.setArg(args++, nb_ghosts); 
    k_ghost_hash.setArg(args++, cl_ghosts.getDevicePtr()); // positions + other variables
	k_ghost_hash.setArg(args++, cl_sort_hashes.getDevicePtr());
	k_ghost_hash.setArg(args++, cl_sort_indices.getDevicePtr());
	k_ghost_hash.setArg(args++, cl_GridParams.getDevicePtr());
    k_ghost_hash.setArg(args++, clf_debug.getDevicePtr());
	k_ghost_hash.setArg(args++, cli_debug.getDevicePtr());


}

void SPH::loadGhostDataStructures()
{
    printf("create ghost datastructures kernel\n");
    std::string path(SPH_CL_SOURCE_DIR);
	//path = path + "/ghost_datastructures_cl.cl";
	path = path + "/ghost_datastructures_cl.cl";
    //k_ghost_datastructures = Kernel(ps->cli, path, "ghost_datastructures");
    k_ghost_datastructures = Kernel(ps->cli, path, "ghost_datastructures");
    printf("kernel made, set args\n");

    int iarg = 0;
	k_ghost_datastructures.setArg(iarg++, nb_ghosts);
	k_ghost_datastructures.setArg(iarg++, cl_ghosts.getDevicePtr());
    k_ghost_datastructures.setArg(iarg++, cl_ghosts_sorted.getDevicePtr());
	k_ghost_datastructures.setArg(iarg++, cl_sort_hashes.getDevicePtr());
	k_ghost_datastructures.setArg(iarg++, cl_sort_indices.getDevicePtr());
	k_ghost_datastructures.setArg(iarg++, cl_SPHParams.getDevicePtr());
	//k_ghost_datastructures.setArg(iarg++, cl_GridParamsScaled->getDevicePtr());
    
    int workSize = 64;
	int nb_bytes = (workSize+1)*sizeof(int);
    k_ghost_datastructures.setArgShared(iarg++, nb_bytes);

}

void SPH::ghost_hash()
// Generate hash list: stored in cl_sort_hashes
{

	int ctaSize = 128; // work group size
	// Hash based on unscaled data
    printf("ghost hash set arg\n");
    k_ghost_hash.setArg(0, nb_ghosts);
    printf("ghost hash execute\n"); 
	k_ghost_hash.execute(max_num, ctaSize);

	ps->cli->queue.finish();
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


void SPH::build_ghost_datastructures()
// Generate hash list: stored in cl_sort_hashes
{


    /*
    int nbc = 20;
    std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
    //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    for(int i = 0; i < nbc; i++)
    {
        printf("sh[%d] %d\n", i, sh[i]);
    }
    */


    //printf("about to data structures\n");
	int workSize = 64; // work group size
    try
    {
	    k_ghost_datastructures.execute(max_num, workSize);
    }
    catch (cl::Error er) {
        printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
	
    ps->cli->queue.finish();
    
    int nbc = 320;
    //std::vector<float4> g = cl_ghosts.copyToHost(300, nbc);
    //std::vector<float4> gs = cl_ghosts_sorted.copyToHost(300, nbc);
    std::vector<float4> g = cl_ghosts.copyToHost(nbc);
    std::vector<float4> gs = cl_ghosts_sorted.copyToHost(nbc);

    for(int i = 313; i < nbc; i++)
    {
        printf("ghosts[%d].xyz %f %f %f  || ghosts_sorted[%d].xyz %f %f %f\n", i, g[i].x, g[i].y, g[i].z, i, gs[i].x, gs[i].y, gs[i].z);
    }


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


//----------------------------------------------------------------------

void SPH::printGhostHashDiagnostics()
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
