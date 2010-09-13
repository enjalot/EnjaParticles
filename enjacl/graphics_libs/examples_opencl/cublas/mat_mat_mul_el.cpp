#include <timege.h>
extern GE::Time clock_mat_mul_cpu;
extern GE::Time clock_mat_mul_gpu;

// put after include of kh file

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <string>
#include "cl.h"
using namespace std;


void mat_mat_mul_el(cl_mem res, cl_mem a, cl_mem b, int nx, int ny, int nz)
{
	static cll_Program* prog = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_mat_mul_cpu.begin();

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/mat_mat_mul_el_kernel.cl";
		printf("prog == 0\n");
		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	// <<< STRONG SLOWDOWN: factor 6
	// timers inde prog->addKernel indicate no slowdown!!!
	// Yet the call itself has very strong slowdown!!! WHY?
	cll_Kernel kern = prog->addKernel("mat_mat_mul_el_kernel"); 

	kern.setArg(res, 0);
	kern.setArg(a, 1);
	kern.setArg(b, 2);

	//size_t global[2];
	//size_t local[2];

	size_t global = nx*ny*nz;
	//size_t local = 128; // GPU: 5.6 ms (on mac)
	//local = 2; // GPU: 30ms 
	size_t local = 64; // GPU: 4.4 ms (on mac)
	//local = 32; // GPU: 5 (on mac)

// (32,4,1) (block)

// make sure local[i] is below max work size (64, 128, etc.). 

	clock_mat_mul_gpu.begin();
    cl_event exec = kern.exec(1, &global, &local);

	cl.waitForKernelsToFinish();
	clock_mat_mul_gpu.end();
	clock_mat_mul_cpu.end();
}
//----------------------------------------------------------------------
