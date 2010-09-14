
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

	ArrayOpenCL1D<cl_float4> ar(10,10,10);
	//ArrayOpenCL1D<float>* ar = new ArrayOpenCL1D<float>(10,10,10);

	int iarg = 0;
	kern.setArg(nb_el, iarg++);
	kern.setArg(nb_vars, iarg++);
	printf("sizeof(int*) = %d\n", sizeof(int*));
	kern.setArg(ar.getDevicePtr(), iarg++);
	kern.setArg(cl_vars_unsorted(), iarg++); // CANNOT SET ARGUMENT
	exit(0);
	kern.setArg(cl_vars_sorted(), iarg++);
	//kern.setArg(force, iarg++);
	//kern.setArg(pressure, iarg++);
	//kern.setArg(density, iarg++);
	//kern.setArg(position, iarg++);
	//kern.setArg(force_sorted, iarg++);
	//kern.setArg(pressure_sortd, iarg++);
	//kern.setArg(density_sorted, iarg++);
	//kern.setArg(position_sorted, iarg++);
	exit(0);
	kern.setArg(cl_cell_indices_start(), iarg++);
	kern.setArg(cl_cell_indices_end(),   iarg++);
	exit(0);
	kern.setArg(gp, iarg++);


	size_t global = (size_t) nb_el;
	size_t local = cl.getMaxWorkSize(kern.getKernel());

	cl_event exec = kern.exec(1, &global, &local);
	cl.waitForKernelsToFinish();
}

