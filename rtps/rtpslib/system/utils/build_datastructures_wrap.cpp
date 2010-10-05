#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::buildDataStructures()
{
	static bool first_time = true;

	//printf("buildData\n"); exit(0);

	//printf("ENTER BUILD\n");

	ts_cl[TI_BUILD]->start();

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/datastructures_test.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			//printf("length= %d\n", length);
			std::string strg(src);
        	datastructures_kernel = Kernel(ps->cli, strg, "datastructures");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = datastructures_kernel;

	int workSize = 64;

	// HOW TO DEAL WITH ARGUMENTS

	//printf("BEFORE BUILD ARGS\n");

	#if 0
	// Slowdown by fact of 3
	// Set cl_cell_indices_start to -1 (0xffffffff)
	// INEFFICIENTLY
	int* st = cl_cell_indices_start->getHostPtr();
	for (int i=0; i < grid_size; i++) {
		st[i] = 0xffffffff;
	}
	cl_cell_indices_start->copyToDevice();
	#endif
		

	int iarg = 0;
	kern.setArg(iarg++, cl_vars_unsorted->getDevicePtr());
	kern.setArg(iarg++, cl_vars_sorted->getDevicePtr());
	kern.setArg(iarg++, cl_sort_hashes->getDevicePtr());
	kern.setArg(iarg++, cl_sort_indices->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end->getDevicePtr());
	kern.setArg(iarg++, cl_GridParams->getDevicePtr());


	// local memory
	int nb_bytes = (workSize+1)*sizeof(int);
    kern.setArgShared(iarg++, nb_bytes);


	//printf("AFTER BUILD ARGS\n");

	int err;
	//printf("EXECUTE BUILD\n");
   	kern.execute(nb_el, workSize); 
	//printf("AFTER EXECUTE BUILD\n");

	//printBuildDiagnostics();
	//printf("after build diagnostics\n");
	//exit(0);

    ps->cli->queue.finish();
	ts_cl[TI_BUILD]->end();
	//exit(0);
}
//----------------------------------------------------------------------
void GE_SPH::printBuildDiagnostics()
{
#if 1
	// should try with and without exceptions. 
	// DATA SHOULD STAY ON THE GPU
	try {
		cl_vars_unsorted->copyToHost();
		cl_vars_sorted->copyToHost();
		cl_sort_indices->copyToHost();
		//cl_vars_sort_indices->copyToDevice(); // not defineed
		//exit(0);
	} catch(cl::Error er) {
        printf("1 ERROR(buildDatastructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	float4* us =  cl_vars_unsorted->getHostPtr();
	float4* so =  cl_vars_sorted->getHostPtr();
	int* si =  cl_sort_indices->getHostPtr();

	// PRINT OUT UNSORTED AND SORTED VARIABLES
	#if 1
	//for (int k=0; k < nb_vars; k++) {
	for (int k=1; k < 2; k++) {
		printf("================================================\n");
		printf("=== VARIABLE: %d ===============================\n", k);
	for (int i=0; i < nb_el; i++) {
		float4 uss = us[i+k*nb_el];
		//printf("uss= %f, %f, %f, %f\n", uss.x, uss.y, uss.z, uss.w);
		float4 soo = so[i+k*nb_el];
		//printf("soo= %f, %f, %f, %f\n", soo.x, soo.y, soo.z, soo.w);
		printf("[%d]: cl_sort_indices: %d, vars_unsorted: (%f, %f), vars_sorted: (%f, %f)\n", i, si[i], uss.x, uss.y, soo.x, soo.y);
		//printf("[%d]: %d \n", i, si[i]);
	}
	}
	printf("===============================================\n");
	printf("===============================================\n");
	#endif


	try {
		// PRINT OUT START and END CELL INDICES
		cl_cell_indices_start->copyToHost();
		cl_cell_indices_end->copyToHost();
	} catch(cl::Error er) {
        printf("2 ERROR(buildDatastructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	computeCellStartEndGPU();
	//printf("return from BuildDataStructures\n");
}
//----------------------------------------------------------------------
void GE_SPH::computeCellStartEndGPU()
{
	int* is = cl_cell_indices_start->getHostPtr();
	int* ie = cl_cell_indices_end->getHostPtr();

		printf("cell_indices_start, end (GPU)\n");
		int nb_particles = 0;
		for (int i=0; i < grid_size; i++) {
			int nb = ie[i] - is[i];
			if (nb > 0 && is[i] != -1) {
				nb_particles += nb;
				//printf("nb= %d\n", nb);
			}
			//if (is[i] != -1) {
				//printf("(GPU) [%d]: indices_start: %d, indices_end: %d, nb pts: %d\n", i, is[i], ie[i], nb);
			//}
		}
		if (nb_particles != nb_el) {
			printf("nb particles: %d\n", nb_particles);
			printf("nb_el: %d\n", nb_el);
			printf("(computeCellStartEndGPU) (count != nb_el)\n");
			exit(1);
		}
}
//----------------------------------------------------------------------
#endif


} // namespace
