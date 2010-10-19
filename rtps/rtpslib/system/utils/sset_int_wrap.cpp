
/* This file contains the implementation of the BLAS-1 function scopy */

#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//   xdst[i] <--- val
//----------------------------------------------------------------------
void GE_SPH::sset(int n, int val, cl_mem xdst)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/sset_int.cl";
			int length;
			char* src = file_contents(path.c_str(), &length);
			//printf("src= %s\n", src); exit(0);
			std::string strg(src);
        	sset_int_kernel = Kernel(ps->cli, strg, "sset_int");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}


	Kernel kern = sset_int_kernel;

	//printf("n= %d, val= %d\n", n, val); exit(0);
	kern.setArg(0, n);
	kern.setArg(1, val);
	kern.setArg(2, xdst);

	size_t global = (size_t) n;
	size_t local = 128; //cl.getMaxWorkSize(kern.getKernel());

	kern.execute(global, local);
	ps->cli->queue.finish();
}
//----------------------------------------------------------------------

} // namespace
