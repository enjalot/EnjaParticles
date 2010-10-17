#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------

void GE_SPH::hash()
// Generate hash list: stored in cl_sort_hashes
{
	static bool first_time = true;

	ts_cl[TI_HASH]->start();

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/uniform_hash.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	hash_kernel = Kernel(ps->cli, strg, "hash");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = hash_kernel;
	float4* cells = cl_cells->getHostPtr();
	int ctaSize = 128; // work group size

	#if 0
	cl_vars_unsorted->copyToHost();
	ps->cli->queue.finish();
	float4* vp = cl_vars_unsorted->getHostPtr();
	//printf("v= %d\n", v);
	for (int i=0; i < nb_el; i++) {
		float4* v = vp;
		printf("==================\n");
		printf("0: v[%d]= %f, %f, %f, %f\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
		v += nb_el;
		printf("1: v[%d]= %f, %f, %f, %f\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
		v += nb_el;
		printf("2: v[%d]= %f, %f, %f, %f\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
	}
	exit(0);
	#endif

	#if 0
	cl_sort_hashes->copyToHost();
	cl_sort_indices->copyToHost();
	cl_cell_indices_start->copyToHost();
	ps->cli->queue.finish();

	int* v1 = cl_sort_hashes->getHostPtr();
	int* v2 = cl_sort_indices->getHostPtr();
	int* v3 = cl_cell_indices_start->getHostPtr();
	//printf("v= %d\n", v);
	for (int i=0; i < nb_el; i++) {
		printf("0: v123[%d]= %d, %d, %d\n", i, v1[i], v2[i], v3[i]);
	}

	cl_GridParams->copyToHost();
	GridParams* gp = cl_GridParams->getHostPtr();
	gp->print();
	exit(0);
	#endif

	// Hash based on unscaled data

	kern.setArg(0, cl_vars_unsorted->getDevicePtr()); // positions + other variables
	kern.setArg(1, cl_sort_hashes->getDevicePtr());
	kern.setArg(2, cl_sort_indices->getDevicePtr());
	kern.setArg(3, cl_cell_indices_start->getDevicePtr());
	kern.setArg(4, cl_GridParams->getDevicePtr());
	//kern.setArg(4, cl_GridParamsScaled->getDevicePtr());
	kern.setArg(5, clf_debug->getDevicePtr());
	kern.setArg(6, cli_debug->getDevicePtr());

	//printf("nb_el= %d\n", nb_el);
	kern.execute(nb_el,ctaSize);
	//exit(0);

	ps->cli->queue.finish();
	ts_cl[TI_HASH]->end();

	printHashDiagnostics();
}
//----------------------------------------------------------------------
void GE_SPH::printHashDiagnostics()
{
#if 1
	printf("***** PRINT hash diagnostics ******\n");
	cl_sort_hashes->copyToHost();
	cl_sort_indices->copyToHost();
	cl_cells->copyToHost();
	cli_debug->copyToHost();
	clf_debug->copyToHost();
	cl_GridParams->copyToHost();

	GridParams& gp = *cl_GridParams->getHostPtr();
	gp.print();

	//cli_debug->copyToHost();

	for (int i=0; i < nb_el; i++) {  // only first 4096 are ok. WHY? 
		printf(" cl_sort_hash[%d] %u, cl_sort_indices[%d]: %u\n", i, (*cl_sort_hashes)[i], i, (*cl_sort_indices)[i]);
		printf("cli_debug: %d, %d, %d\n", (*cli_debug)[i].x, (*cli_debug)[i].y, (*cli_debug)[i].z);
		printf("clf_debug: %f, %f, %f\n", (*clf_debug)[i].x, (*clf_debug)[i].y, (*clf_debug)[i].z);
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

} // namespace
