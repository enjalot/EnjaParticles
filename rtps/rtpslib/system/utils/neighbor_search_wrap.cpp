#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------

//----------------------------------------------------------------------
void GE_SPH::neighborSearch(int which)
{
	static bool first_time = true;

	// which == 0 : density update
	// which == 1 : pressure update

	if (which == 0) ts_cl[TI_DENS]->start();
	if (which == 1) ts_cl[TI_PRES]->start();
	//ts_cl[TI_NEIGH]->start();

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

	#if 1
	float4* fclf = clf_debug->getHostPtr();
	int4*   icli = cli_debug->getHostPtr();

	// Need to set this to zero
	for (int i=0; i < nb_el; i++) { 
		fclf[i].x = 0.; fclf[i].y = 0.; fclf[i].z = 0.; fclf[i].w = 0.;
		icli[i].x = 0; icli[i].y = 0; icli[i].z = 0; icli[i].w = 0;
	}
	clf_debug->copyToDevice();
	cli_debug->copyToDevice();
	#endif

	FluidParams* fp = cl_FluidParams->getHostPtr();
	fp->choice = which;
	cl_FluidParams->copyToDevice();
	int iarg = 0;
	
	//kern.setArg(iarg++, nb_el);
	//kern.setArg(iarg++, nb_vars);
	kern.setArg(iarg++, cl_vars_sorted->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_start->getDevicePtr());
	kern.setArg(iarg++, cl_cell_indices_end->getDevicePtr());
	kern.setArg(iarg++, cl_GridParams->getDevicePtr());
	kern.setArg(iarg++, cl_FluidParams->getDevicePtr());
	kern.setArg(iarg++, cl_params->getDevicePtr());

	// ONLY IF DEBUGGING
	kern.setArg(iarg++, clf_debug->getDevicePtr());
	kern.setArg(iarg++, cli_debug->getDevicePtr());


	size_t global = (size_t) nb_el;
	int local = 128;

	kern.execute(nb_el, local);

	ps->cli->queue.finish();
	//ts_cl[TI_NEIGH]->end();
	if (which == 0) ts_cl[TI_DENS]->end();
	if (which == 1) ts_cl[TI_PRES]->end();


	#if 0
if (which == 1) {
	printf("which == 1 *** \n");
	clf_debug->copyToHost();
	cli_debug->copyToHost();

	for (int i=0; i < nb_el; i++) { 
	//for (int i=0; i < 10; i++) { 
		printf("----------------------------\n");
		printf("clf[%d]= %f, %f, %f, %f\n", i, fclf[i].x, fclf[i].y, fclf[i].z, fclf[i].w);
		printf("cli[%d]= %d, %d, %d, %d\n", i, icli[i].x, icli[i].y, icli[i].z, icli[i].w);
	}
	//exit(0);
}
	#endif
	//if (which ==1) exit(0);
}
//----------------------------------------------------------------------

} // namespace
