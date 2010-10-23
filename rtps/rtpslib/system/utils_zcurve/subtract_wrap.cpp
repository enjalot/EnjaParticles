#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
// ADD ARGUMENTS
void GE_SPH::subtract()
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/subtract.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			//printf("length= %d\n", length);
			std::string strg(src);
        	subtract_kernel = Kernel(ps->cli, strg, "subtract");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = subtract_kernel;


	GridParamsScaled* gps = cl_GridParamsScaled->getHostPtr();

	// global must be an integer multiple of work_size
	size_t global = (size_t) gps->nb_points;
	int work_size = 128;
	int ii = global / work_size;
	int global1 = ii*work_size;
	if (global1 != global) global = global1 + work_size;

	int iarg = 0;
	kern.setArg(iarg++, gps->nb_points);
	kern.setArg(iarg++, cl_cell_indices_end->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_nb->getDevicePtr());

   	kern.execute(global1, work_size); 
    ps->cli->queue.finish();

	#if 0
	// subtract works ok
	cl_cell_indices_start->copyToHost();
	cl_cell_indices_end->copyToHost();
	cl_cell_indices_nb->copyToHost();
	int* st = cl_cell_indices_start->getHostPtr();
	int* en = cl_cell_indices_end->getHostPtr();
	int* nb = cl_cell_indices_nb->getHostPtr();
	for (int i=0; i < gps->nb_points; i++) {
		if (nb[i] <= 0) continue;
		printf("(%d), st, en, nb= %d, %d, %d\n", i, st[i], en[i], nb[i]);
	}
	exit(0);
	#endif
}
//----------------------------------------------------------------------
}
