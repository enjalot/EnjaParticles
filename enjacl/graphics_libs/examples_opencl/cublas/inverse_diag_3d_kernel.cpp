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

#include <string>
#include "cl.h"
using namespace std;

// temporary
#include <array_opencl_1d.h>


//----------------------------------------------------------------------
void inverseDiag3DKernel(cl_mem a, int nx, int ny, int nz, int nb_zblocks, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM)
{
	static cll_Program* prog = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_inv_mat_vec_cpu.begin();

	size_t global[3];
	size_t local[3];

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/inverse_tex_add.cl";
		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	local[0] = 16;
	local[1] = 4;
	local[2] = 2;

	global[0] = nx; //  / local[0];
	global[1] = ny; //  / local[1];
	global[2] = local[2];


	cll_Kernel kern = prog->addKernel("inverse_diag_3d_kernel"); 

	kern.setArg(a, 0);
	kern.setArg(nx, 1); 
	kern.setArg(ny, 2);
	kern.setArg(nz, 3);
	kern.setArg(nb_zblocks, 4);
	kern.setArg(KM, 5);
	kern.setArg(KP, 6);
	kern.setArg(LM, 7);
	kern.setArg(LP, 8);
	kern.setArg(MM, 9);
	kern.setArg(MP, 10);


// make sure local[i] is below max work size (64, 128, etc.). 

	clock_inv_mat_vec_gpu.begin();

    cl_event exec = kern.exec(3, &global[0], &local[0]);
	cl.waitForKernelsToFinish();

	clock_inv_mat_vec_gpu.end();
	clock_inv_mat_vec_cpu.end();
}
//----------------------------------------------------------------------
void invAddDirichlet(cl_mem a, int nx, int ny, int nz, int nb_zblocks, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM)
// Same as inverseDiag3DKernel, but with conditionals to avoid treatment of first and last planes in 
// all 3 directions (single layer boundaries included in 
{
	static cll_Program* prog = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_inv_mat_vec_cpu.begin();

	size_t global[3];
	size_t local[3];

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/inverse_tex_add_dirichlet.cl";
		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	local[0] = 16;
	local[1] = 4;
	local[2] = 2;

	global[0] = nx; //  / local[0];
	global[1] = ny; //  / local[1];
	global[2] = local[2];


	cll_Kernel kern = prog->addKernel("inverse_tex_add_dirichlet"); 

	kern.setArg(a, 0);
	kern.setArg(nx, 1); 
	kern.setArg(ny, 2);
	kern.setArg(nz, 3);
	kern.setArg(nb_zblocks, 4);
	kern.setArg(KM, 5);
	kern.setArg(KP, 6);
	kern.setArg(LM, 7);
	kern.setArg(LP, 8);
	kern.setArg(MM, 9);
	kern.setArg(MP, 10);


// make sure local[i] is below max work size (64, 128, etc.). 

	clock_inv_mat_vec_gpu.begin();

    cl_event exec = kern.exec(3, &global[0], &local[0]);
	cl.waitForKernelsToFinish();

	clock_inv_mat_vec_gpu.end();
	clock_inv_mat_vec_cpu.end();
}
//----------------------------------------------------------------------
