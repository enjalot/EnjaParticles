#include "../GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::compactify(BufferGE<int>& cl_orig, BufferGE<int>&  cl_compact, 
	BufferGE<int>& cl_processorCounts, BufferGE<int>& cl_processorOffsets)
{
	static bool first_time = true;
	static int work_size = 0;
	int nb_warps;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			//path = path + "/block_scan_block64_cl.cl";
			path = path + "/compactify_cl.cl";
			work_size = 32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_kernel = Kernel(ps->cli, strg, "compactifyArrayKernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	//BufferGE<int> cl_processorCounts(ps->cli, work_size);
	//BufferGE<int> cl_processorOffsets(ps->cli, work_size);

	nb_warps = work_size / 32;

	Kernel kern = compactify_kernel;
	kern.setProfiling(true);

	//FluidParams* fp = cl_FluidParams->getHostPtr();
	//cl_FluidParams->copyToDevice();
	//GridParamsScaled* gps = cl_GridParamsScaled->getHostPtr();

	
	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getDevicePtr());
	kern.setArg(iarg++, cl_processorCounts.getDevicePtr());
	kern.setArg(iarg++, cl_processorOffsets.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getSize());
	#endif

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	
	size_t nb_blocks = cl_orig.getSize() / work_size;
	printf("nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("orig size: %d\n", cl_orig.getSize());
	if ((work_size*nb_blocks) != cl_orig.getSize()) nb_blocks++;

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY]->end();
}
//----------------------------------------------------------------------
void GE_SPH::compactifyDown(BufferGE<int>& cl_orig, BufferGE<int>&  cl_compact,
	BufferGE<int>& cl_processorCounts, BufferGE<int>& cl_processorOffsets)
{
	static bool first_time = true;
	static int work_size = 0;
	int nb_warps;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			path = path + "/compactify_down_cl.cl";
			work_size = 32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_down_kernel = Kernel(ps->cli, strg, "compactifyDownKernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	//BufferGE<int> cl_processorCounts(ps->cli, work_size);
	//BufferGE<int> cl_processorOffsets(ps->cli, work_size);

	nb_warps = work_size / 32;

	Kernel kern = compactify_kernel;
	kern.setProfiling(true);

	FluidParams* fp = cl_FluidParams->getHostPtr();
	cl_FluidParams->copyToDevice();

	GridParamsScaled* gps = cl_GridParamsScaled->getHostPtr();

	
	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getDevicePtr());
	kern.setArg(iarg++, cl_processorCounts.getDevicePtr());
	kern.setArg(iarg++, cl_processorOffsets.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getSize());
	#endif

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	
	size_t nb_blocks = cl_orig.getSize() / work_size;
	printf("nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("orig size: %d\n", cl_orig.getSize());
	if ((work_size*nb_blocks) != cl_orig.getSize()) nb_blocks++;

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY]->end();
}
//----------------------------------------------------------------------

} // namespace
