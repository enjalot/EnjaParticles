#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::blockScan(int which)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/block_scan_cl.cl";
			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	block_scan_kernel = Kernel(ps->cli, strg, "block_scan");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = block_scan_kernel;

	FluidParams* fp = cl_FluidParams->getHostPtr();
	fp->choice = which;
	cl_FluidParams->copyToDevice();

	GridParamsScaled* gps = cl_GridParamsScaled->getHostPtr();
	
	int iarg = 0;
	//kern.setArg(iarg++, gps->nb_points);
	kern.setArg(iarg++, cl_vars_sorted->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_nb->getDevicePtr());
	kern.setArg(iarg++, cl_hash_to_grid_index->getDevicePtr());
	kern.setArg(iarg++, cl_cell_offset->getDevicePtr());
	kern.setArg(iarg++, cl_params->getDevicePtr());
	kern.setArg(iarg++, cl_GridParamsScaled->getDevicePtr());

	#if 1
	cl_cell_offset->copyToHost();
	int4* cc = cl_cell_offset->getHostPtr();
	for (int i=0; i < 27; i++) {
		cc[i].print("offset");
	}
	//exit(0);
	#endif

	cl_cell_offset->copyToDevice();


	// local memory
	// space for 4 variables of 4 bytes (float) for (27+32) particles
	int nb_bytes = 1024 * sizeof(float);
    kern.setArgShared(iarg++, nb_bytes);

	// ONLY IF DEBUGGING
	kern.setArg(iarg++, clf_debug->getDevicePtr());
	kern.setArg(iarg++, cli_debug->getDevicePtr());

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	size_t nb_blocks = (size_t) gps->nb_points;
	int work_size = 32;  // probably inefficient. Increase to 64 in the future perhaps
	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;

	//cl_GridParamsScaled->copyToHost();
	//printf("nb points in grid: %d\n", gps->nb_points);
	//exit(0);

	kern.execute(global, work_size);

	ps->cli->queue.finish();

	#if 0
	// subtract works ok
	cl_cell_indices_start->copyToHost();
	cl_cell_indices_end->copyToHost();
	cl_cell_indices_nb->copyToHost();
	int* st = cl_cell_indices_start->getHostPtr();
	int* en = cl_cell_indices_end->getHostPtr();
	int* nb = cl_cell_indices_nb->getHostPtr();
	for (int i=0; i < nb_el; i++) {
		printf("(%d), st, en, nb= %d, %d, %d\n", i, st[i], en[i], nb[i]);
	}
	exit(0);
	#endif


	#if 1
	printf("============================================\n");

	clf_debug->copyToHost();
	cli_debug->copyToHost();
	float4* fclf = clf_debug->getHostPtr();
	int4*   icli = cli_debug->getHostPtr();
	cl_cell_indices_nb->copyToHost();
	int* nb = cl_cell_indices_nb->getHostPtr();

	cl_index_neigh->copyToHost();
	int* n = cl_index_neigh->getHostPtr();

	int count=0;

	for (int i=0; i < gps->nb_points; i++) { 
		if (nb[i] <= 0) continue;
		count += nb[i];
		printf("----------------------------\n");
		printf("clf[%d]= %f, %f, %f, %f\n", i, fclf[i].x, fclf[i].y, fclf[i].z, fclf[i].w);
		printf("cli[%d]= %d, %d, %d, %d\n", i, icli[i].x, icli[i].y, icli[i].z, icli[i].w);
		printf("nb[%d]= %d\n", i, nb[i]);
		//printf("index(%d): (%d)", i, icli[i].x); 
		//int max = icli[i].x < 50 ? icli[i].x : 50;
		//for (int j=0; j < icli[i].x; j++) {
			//printf("%d, ", n[j+50*i]);
		//}
		//printf("\n");
	}
	printf("count= %d, tot nb particles: %d\n", count, nb_el);
	printf("nb grid points: %d\n", gps->nb_points);

	gps->print();
	exit(0);
	#endif
}
//----------------------------------------------------------------------

} // namespace
