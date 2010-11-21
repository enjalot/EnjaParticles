#include "../GE_SPH.h"

#include <string>
using namespace std;

// nb warps per block = MULT
#define MULT 4
#define MULT_MIDDLE 4

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::compactify(BufferGE<int>& cl_orig, BufferGE<int>&  cl_compact, 
	BufferGE<int>& cl_processorCounts, BufferGE<int>& cl_processorOffsets)
{
	static bool first_time = true;
	static int work_size = 0;
	int nb_warps;

	if (first_time) {
		// more useful routine, not yet written
		//setupKernel(ps->cli, path, kernel, kernel_name, work_size, block_size); 
		//if (!kernel.is_compiled) {
		//	kernel_setupKernel(ps->cli, path, kernel, kernel_name, work_size, block_size); 
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			//path = path + "/block_scan_block64_cl.cl";
			path = path + "/compactify_cl.cl";
			// blocks of size 256 is best on GTX 330M
			work_size = MULT*32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_kernel = Kernel(ps->cli, strg, "compactifyArrayKernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	nb_warps = work_size / 32;

	Kernel kern = compactify_kernel;
	kern.setProfiling(true);

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
	printf("compactify::nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("compactify::orig size: %d\n", cl_orig.getSize());
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
			work_size = MULT*32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_down_kernel = Kernel(ps->cli, strg, "compactifyDownKernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	nb_warps = work_size / 32;

	Kernel kern = compactify_down_kernel;
	kern.setProfiling(true);
	
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
	
	GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());
	size_t nb_blocks = cl_orig.getSize() / work_size;

	// in fact: nb blocks will eventually be: nb_points / block_size)
	if (nb_blocks > (gps.nb_points/32)) {
		printf("nb_blocks is greater than %d, equal to memory size \n", gps.nb_points/32);
		printf("  allocated to processorCount and processorOffset\n");
		exit(0);
	}
	printf("nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("orig size: %d\n", cl_orig.getSize());
	if ((work_size*nb_blocks) != cl_orig.getSize()) nb_blocks++;

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_DOWN]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_DOWN]->end();
}
//----------------------------------------------------------------------
void GE_SPH::compactifyMiddle(BufferGE<int>& cl_processorCounts, 
		BufferGE<int>& cl_processorOffsets,
		BufferGE<int>& cl_temp_sums)
{
	static bool first_time = true;
	static int work_size = 0;
	int nb_warps;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			path = path + "/compactify_middle_cl.cl";
			work_size = MULT_MIDDLE*32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_middle_kernel = Kernel(ps->cli, strg, "compactifyMiddleKernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	nb_warps = work_size / 32;

	Kernel kern = compactify_middle_kernel;
	kern.setProfiling(true);

	
	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_processorCounts.getDevicePtr());
	kern.setArg(iarg++, cl_processorOffsets.getDevicePtr());
	kern.setArg(iarg++, cl_temp_sums.getDevicePtr());
	kern.setArg(iarg++, 2048); // hardcoded!!
	#endif

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	
	GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());
	//size_t nb_blocks = cl_processorOffsets.getSize() / work_size;
	// 2048: nb blocks in compactify
	size_t nb_blocks = 2048 / work_size;

	if (cl_temp_sums.getSize() < nb_blocks) {
		printf("middle: cl_temp_sums size is too small\n");
		exit(0);
	}

	// in fact: nb blocks will eventually be: nb_points / block_size)
	if (nb_blocks > (gps.nb_points/32)) {
		printf("nb_blocks is greater than %d, equal to memory size \n", gps.nb_points/32);
		printf("  allocated to processorCount and processorOffset\n");
		exit(0);
	}
	printf("middle: nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("middle: orig size: %d\n", cl_processorOffsets.getSize());
	if ((work_size*nb_blocks) != 2048) {
		nb_blocks++;
		printf("middle: adjusted block size\n");
	}

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_MIDDLE]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_MIDDLE]->end();

	// printout

	#if 1
	cl_processorOffsets.copyToHost();
	cl_processorCounts.copyToHost();
	int* of = cl_processorOffsets.getHostPtr();
	int* co = cl_processorCounts.getHostPtr();
	for (int i=0; i < global; i++) {
		printf("count,offset[%d]= %d, %d\n", i, co[i], of[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	//exit(0);
	#endif
}
//----------------------------------------------------------------------
void GE_SPH::sumScanSingleBlock(BufferGE<int>& cl_input, 
		                        BufferGE<int>& cl_output)
{
	static bool first_time = true;
	static int work_size = 0;
	int nb_warps;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			path = path + "/sum_scan_single_block_cl.cl";
			work_size = MULT_MIDDLE*32;  // SINGLE WARP

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	sum_scan_single_block_kernel = Kernel(ps->cli, strg, "sumScanSingleBlock");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	nb_warps = work_size / 32;


	Kernel kern = sum_scan_single_block_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_output.getDevicePtr());
	kern.setArg(iarg++, cl_input.getDevicePtr());
	kern.setArg(iarg++, 128);  // hardcoded (nb elem = 1 block)
	#endif

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	
	GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());
	// 2048: nb blocks in compactify
	size_t nb_blocks = cl_input.getSize() / work_size;

	if (cl_input.getSize() < nb_blocks) {
		printf("scan_single_block: cl_temp_sums size is too small\n");
		exit(0);
	}


	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;
	printf("scan_single_block: nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("scan_single_block: orig size: %d\n", cl_input.getSize());
	printf("scan_single_block: nb_elem= %d\n", cl_input.getSize());
	printf("scan_single_block: global= %d\n", global);


	ps->cli->queue.finish();
	ts_cl[TI_SCAN_SUM_SINGLE]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_SCAN_SUM_SINGLE]->end();

	// printout

	#if 0
	cl_input.copyToHost();
	cl_output.copyToHost();
	int* in = cl_input.getHostPtr();
	int* ou = cl_output.getHostPtr();
	for (int i=0; i < global; i++) {
		printf("input,output[%d]= %d, %d\n", i, in[i], ou[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	exit(0);
	#endif
}
//----------------------------------------------------------------------

} // namespace
