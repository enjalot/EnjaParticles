
/* This file contains the implementation of the BLAS-1 function scopy */

#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::scopy(int n, cl_mem xsrc, cl_mem ydst)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);
			path = path + "/scopy.cl";
			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	scopy_kernel = Kernel(ps->cli, strg, "scopy");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}


	Kernel kern = scopy_kernel;

	#if 0
    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return;
    }
	#endif
    

	kern.setArg(0, n);
	kern.setArg(1, xsrc);
	kern.setArg(2, ydst);

	size_t global = (size_t) n;
	size_t local = 128; //cl.getMaxWorkSize(kern.getKernel());

	kern.execute(n, local);
	ps->cli->queue.finish();
}
//----------------------------------------------------------------------

} // namespace
