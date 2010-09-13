
#include <array_opencl_1d.h>
using namespace std;

void EnjaParticles::neighbor_search()
{
	static cll_Program* prog = 0;
	CL cl(true);
	//cl.waitForKernels ToFinish();
	printf("junk\n");

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		//path = path + "/neighbor_search.cl";
		path = path + "/uniform_grid_utils.cl";
		printf("prog == 0\n");
		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	cll_Kernel kern = prog->addKernel("K_SumStep1");

	//kern.setArg();
}
