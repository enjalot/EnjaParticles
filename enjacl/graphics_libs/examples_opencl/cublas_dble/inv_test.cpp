#include <timege.h>
extern GE::Time clock_inv_mat_vec_cpu;
extern GE::Time clock_inv_mat_vec_gpu;

// put after include of kh file

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
// temporary
#include <array_opencl_1d.h>

#include "float_type.h"
#include "cl.h"

using namespace std;



//----------------------------------------------------------------------
void inverseTest(cl_mem a, cl_mem b, int nx, int ny, int nz)
{
	static cll_Program* prog = 0;
	printf("enter inv_test\n");

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_inv_mat_vec_cpu.begin();

	size_t global[1];
	size_t local[1];

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/inv_test.cl";

        string double_option;
#ifdef DOUBLE
        double_option = " -DDOUBLE ";
#else
        double_option = "";
#endif

        string graphic_libs_dir = getenv("GRAPHIC_LIBS_HOME");
        string options = double_option + "-I" + graphic_libs_dir +
"/include";
        cl.setCompilerOptions(options.c_str());

		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	local[0] = 64;
	//local[1] = 4;
	//local[2] = 2;

	global[0] = nx*ny*nz; //  / local[0];
	//global[1] = ny; //  / local[1];
	//global[2] = local[2];


	cll_Kernel kern = prog->addKernel("inverse_diag_3d_kernel"); 

	kern.setArg(a, 0);
	kern.setArg(b, 1);

// make sure local[i] is below max work size (64, 128, etc.). 

	clock_inv_mat_vec_gpu.begin();

	printf("before exec\n");
    cl_event exec = kern.exec(1, &global[0], &local[0]);
	printf("after exec\n");
	cl.waitForKernelsToFinish();

	clock_inv_mat_vec_gpu.end();
	clock_inv_mat_vec_cpu.end();
	printf("exit inv*cpp\n");
}
//----------------------------------------------------------------------
