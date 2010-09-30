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

	kern.setArg(0, cl_cells->getDevicePtr());
	kern.setArg(1, cl_sort_hashes->getDevicePtr());
	kern.setArg(2, cl_sort_indices->getDevicePtr());
	kern.setArg(3, cl_GridParams->getDevicePtr());
	//kern.setArg(4, clf_debug->getDevicePtr());
	//kern.setArg(5, cli_debug->getDevicePtr());

	kern.execute(nb_el,ctaSize);

	ps->cli->queue.finish();
	ts_cl[TI_HASH]->end();

	//printHashDiagnostics();
}
//----------------------------------------------------------------------
void GE_SPH::printHashDiagnostics()
{
#if 1
	printf("***** PRINT hash diagnostics ******\n");
	cl_sort_hashes->copyToHost();
	cl_sort_indices->copyToHost();
	cl_cells->copyToHost();
	cl_GridParams->copyToHost();
	GridParams& gp = *cl_GridParams->getHostPtr();
	gp.grid_size.print("grid size (domain dimensions)"); // domain dimensions
	gp.grid_delta.print("grid delta (cell size)"); // cell size
	gp.grid_min.print("grid min");
	gp.grid_max.print("grid max");
	gp.grid_res.print("grid res (nb points)"); // number of points
	gp.grid_delta.print("grid delta");
	gp.grid_inv_delta.print("grid inv delta");

	//cli_debug->copyToHost();

	for (int i=0; i < nb_el; i++) {  // only first 4096 are ok. WHY? 
		printf(" cl_sort_hash[%d] %u, cl_sort_indices[%d]: %u\n", i, (*cl_sort_hashes)[i], i, (*cl_sort_indices)[i]);

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
		//printf("cli_debug: %d, %d, %d\n", cli_debug[i].x, cli_debug[i].y, cli_debug[i].z);
		//printf("---------------------------\n");
		#endif
	}
#endif
}
//----------------------------------------------------------------------

} // namespace
