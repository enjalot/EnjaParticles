#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------

//----------------------------------------------------------------------
void GE_SPH::neighbor_search()
{
	static bool first_time = true;

	ts_cl[TI_NEIGH]->start();

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/uniform_grid_utils.cl";
			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	step1_kernel = Kernel(ps->cli, strg, "K_SumStep1");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = step1_kernel;

	int iarg = 0;

	kern.setArg(iarg++, nb_el);
	kern.setArg(iarg++, nb_vars);
	kern.setArg(iarg++, cl_vars_unsorted->getDevicePtr());
	kern.setArg(iarg++, cl_vars_sorted->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end->getDevicePtr());
	kern.setArg(iarg++, cl_GridParams->getDevicePtr());
	kern.setArg(iarg++, cl_FluidParams->getDevicePtr());


	size_t global = (size_t) nb_el;
	int local = 128;

	kern.execute(nb_el, local);
	exit(0);

	ps->cli->queue.finish();
	ts_cl[TI_NEIGH]->end();
}
//----------------------------------------------------------------------

} // namespace
