#include <timege.h>
extern GE::Time clock_mat_vec_cpu;
extern GE::Time clock_mat_vec_gpu;

// put after include of kh file

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include "cl.h"
#include "float_type.h"

using namespace std;

// temporary
#include <array_opencl_1d.h>


void matVecKernel(cl_mem a, cl_mem b, int nx, int ny, int nz, 
    int nb_zblocks, cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM)
{
	static cll_Program* prog = 0;
	static ArrayOpenCL1D<FLOAT>* local_mem = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_mat_vec_cpu.begin();

	size_t global[3];
	size_t local[3];

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/mat_vec.cl";

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
		local[0] = 16;
		local[1] = 4;
		local[2] = 2;
		local_mem = new ArrayOpenCL1D<FLOAT>(local[0]+2, local[1]+2, local[2]+2);
		prog = &progr;
	}

	//size_t global = nx*ny*nz;
	//size_t local = 128; // GPU: 5.6 ms (on mac)
	//local = 2; // GPU: 30ms 
	//size_t local = 64; // GPU: 4.4 ms (on mac)
	//local = 32; // GPU: 5 (on mac)

	local[0] = 16;
	local[1] = 4;
	local[2] = 2;

	global[0] = nx; //  / local[0];
	global[1] = ny; //  / local[1];
	global[2] = local[2];

	//global[0] = 10;
	//global[1] = 10;
	//global[2] = 1;

	int bx = local[0]+2;        // block.x + 2
	int cx = bx*(local[1]+2);   // bx * (block.y + 2)
	int zblocksz = nz / nb_zblocks;

#if 0
	cll_Kernel kern_tst = prog->addKernel("tst_mul"); 
	kern_tst.setArg(a, 0);
	kern_tst.setArg(b, 1);

	size_t ggg[3]; ggg[0] = ggg[1] = 100; ggg[2] = 1;
	size_t lll[3]; lll[0] = lll[1] = 10;  lll[2] = 1;
	//global[0] = global[1] = 64; global[2] = 2;
	//local[0] = local[1] = 10;  local[2] = 1;
	local[0] = 32;
	local[1] = 4;
	local[2] = 4;
	global[0] = 64;
	global[1] = 64;
	global[2] = 8;

	//kern_tst.exec(3, &ggg[0], &lll[0]);    // WORKS
	kern_tst.exec(3, &global[0], &local[0]);
	exit(0);
#endif


	cll_Kernel kern = prog->addKernel("matrix_vec_polar_3d"); 
	//exit(0);

// includes boundary around interior of block
    int block_size = (local[0]+2)*(local[1]+2)*(local[2]+2); 
    int shared_nb_bytes = block_size * sizeof(FLOAT);

	//printf("global: %d, %d, %d\n", global[0], global[1], global[2]);
	//printf("local: %d, %d, %d\n", local[0], local[1], local[2]);
	//exit(0);
	

	kern.setArg(a, 0);
	kern.setArg(b, 1);
	clSetKernelArg(kern.getKernel(), 2, cx*4*sizeof(FLOAT), 0);

	kern.setArg(nx, 3); 
	kern.setArg(ny, 4);
	kern.setArg(nz, 5);
	kern.setArg(nb_zblocks, 6);
	kern.setArg(bx, 7);
	kern.setArg(cx, 8);
	kern.setArg(zblocksz, 9);
	kern.setArg(KM, 10);
	kern.setArg(KP, 11);
	kern.setArg(LM, 12);
	kern.setArg(LP, 13);
	kern.setArg(MM, 14);
	kern.setArg(MP, 15);

	//kern.setArg(local_mem->getDevicePtr(), 15);
	//printf("cx= %d\n", cx);
	//printf("shared mem: %f (kbytes)\n", cx*4.*sizeof(FLOAT)/1024.);


// (32,4,1) (block)

// make sure local[i] is below max work size (64, 128, etc.). 

	clock_mat_vec_gpu.begin();
    cl_event exec = kern.exec(3, &global[0], &local[0]);

	cl.waitForKernelsToFinish();
	clock_mat_vec_gpu.end();
	clock_mat_vec_cpu.end();
}
//----------------------------------------------------------------------
