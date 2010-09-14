
#include <array_opencl_1d.h>
using namespace std;

#include <CL/cl_platform.h>
#include <CL/cl.h>

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

	// SERIOUS PROBLEM with CL++ library. Cannot use with my library when setting
	// cl_mem types. Hard to believe. 

	cll_Kernel kern = prog->addKernel("K_SumStep1");

    //cl::Kernel* ksumstep1;; 
	//cl::Program* psumstep1
	//ksumstep1 = cl::Kernel(psumstep1, "K_SumStep1", &err);

	cl::Kernel kernel();
	kernel.setKernel(kern.getKernel());
	//kernel.setKernel();

	ArrayOpenCL1D<cl_float4> ar(10,10,10);
	//ArrayOpenCL1D<float>* ar = new ArrayOpenCL1D<float>(10,10,10);

	int iarg = 0;
	//kern.setArg(nb_el, iarg++);
	//kern.setArg(nb_vars, iarg++);

	printf("sizeof(int) = %d\n", sizeof(int));
	printf("sizeof(int*) = %d\n", sizeof(int*));
	printf("sizeof(cl_mem) = %d\n", sizeof(cl_mem));
	//kern.setArg(&cl_vars_unsorted, iarg++); // CANNOT SET ARGUMENT
	//kern.setArg(cl_vars_sorted(), iarg++);
	//kern.setArg(cl_cell_indices_start(), iarg++);
	//kern.setArg(cl_cell_indices_end(),   iarg++);
	//kern.setArg(cl_GridParams(), iarg++);

	clSetKernelArg(kern.getKernel(), iarg++, sizeof(int), &nb_el);
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(int), &nb_vars);
	// CREATES AN ERROR ON KERNEL EXECUTION
	// 8 byte addresses, 4-byte ints (in 64 bits addressing mode)
	cl_mem arm = ar.getDevicePtr();
	//clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), arm);
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), cl_vars_unsorted());
	#if 0
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), cl_vars_sorted());
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), cl_cell_indices_start());
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), cl_cell_indices_end());
	clSetKernelArg(kern.getKernel(), iarg++, sizeof(cl_mem), cl_GridParams());
	#endif
	
	// How to wrap my kernel in cl++?
    //err = pos_update_kernel.setArg(2, cl_velocities);   //velocities

	size_t global = (size_t) nb_el;
	size_t local = cl.getMaxWorkSize(kern.getKernel());
	printf("local= %d, global= %d\n", local, global);

	cl_event exec = kern.exec(1, &global, &local);
	cl.waitForKernelsToFinish();
}

