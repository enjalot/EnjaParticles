/* 
Start with 320k grid A, and blocks of size 128
Generate array B of size 320k/128=2048. 
Perform prefix sum on B with blocks of size 64. There are 2048/64=32 blocks
*/
#include "../GE_SPH.h"

#include <string>
using namespace std;

// nb warps per block = MULT
#define MULT 4
#define MULT_MIDDLE 4

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::newCompactifyWrap(BufferGE<int>& cl_orig, BufferGE<int>&  cl_compact, 
	BufferGE<int>& cl_processorCounts, BufferGE<int>& cl_processorOffsets)
{
	static bool first_time = true;
	//static int work_size = 0;

	int work_size = MULT*32;       // 124
	int work_size_1 = MULT*32 / 2; // 64

	int nb_blocks;
	int nb_blocks_1;

	nb_blocks = cl_orig.getSize() / work_size;
	nb_blocks_1 = nb_blocks / work_size_1; 

	printf("work_size, work_size_1= %d, %d\n", work_size, work_size_1);
	printf("nb_blocks, nb_blocks_1= %d, %d\n", nb_blocks, nb_blocks_1);

	BufferGE<int> cl_sum(ps->cli, nb_blocks);
	BufferGE<int> cl_sum_out(ps->cli, nb_blocks);
	// divide by two since in implementation of reduced sum: there are two threads per element
	BufferGE<int> cl_sum_accu(ps->cli, nb_blocks_1/2);
	BufferGE<int> cl_sum_accu_out(ps->cli, nb_blocks_1/2);


	sub1(cl_orig, work_size, nb_blocks, cl_sum);
	sub2(cl_sum, work_size_1, nb_blocks, cl_sum_out,cl_sum_accu);
	sub2Sum(cl_sum_accu, work_size_1, nb_blocks, cl_sum_accu_out); // single block
	sub3(cl_sum_out, work_size_1, nb_blocks, cl_sum_accu_out);
	printf("enter sub4\n");
	sub4(cl_orig, work_size, nb_blocks, cl_sum_out, cl_compact);
}
//----------------------------------------------------------------------
//sub1(cl_orig, work_size, cl_sum);
void GE_SPH::sub1(BufferGE<int>& cl_orig, int work_size, int nb_blks, BufferGE<int>& cl_processorCounts)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			//path = path + "/block_scan_block64_cl.cl";
			path = path + "/compactify_sub1_cl.cl";

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_sub1_kernel = Kernel(ps->cli, strg, "compactifySub1Kernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	Kernel kern = compactify_sub1_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_orig.getDevicePtr());
	kern.setArg(iarg++, cl_processorCounts.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getSize()); // size; 320k
	#endif

	size_t nb_blocks = cl_orig.getSize() / work_size;
	printf("compactify_sub1::nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("compactify_sub1::orig size: %d\n", cl_orig.getSize());
	if ((work_size*nb_blocks) != cl_orig.getSize()) nb_blocks++;

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;
	printf("compactify_sub1::global size: %d\n", global);
	printf("compactify::processcount size: %d\n", cl_processorCounts.getSize());

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB1]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB1]->end();

	#if 0
	cl_processorCounts.copyToHost();
	cl_orig.copyToHost();
	int* in = cl_processorCounts.getHostPtr();
	int* orr = cl_orig.getHostPtr();
	for (int i=0; i < nb_blocks; i++) {
		printf("cl_processorCounts[%d]= %d\n", i, in[i]);
		printf("  orig[%d]= %d\n", i, orr[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	#endif
}
//----------------------------------------------------------------------
void GE_SPH::sub2(BufferGE<int>& cl_sum, int work_size, int nb_blks, BufferGE<int>& cl_sum_out, BufferGE<int>& cl_sum_accu)
{
// input: array size 2048
// block size = work_size = 64
// nb blocks = (2048/2) / 64 = 16
// output: sum_accu = 16

	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			//path = path + "/block_scan_block64_cl.cl";
			path = path + "/compactify_sub2_cl.cl";

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_sub2_kernel = Kernel(ps->cli, strg, "compactifySub2Kernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	Kernel kern = compactify_sub2_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_sum.getDevicePtr());
	kern.setArg(iarg++, cl_sum_out.getDevicePtr());
	kern.setArg(iarg++, cl_sum_accu.getDevicePtr()); 
	kern.setArg(iarg++, cl_sum.getSize()); // size; 2048
	#endif

	// Each thread handles two elements of the block
	size_t nb_blocks = (cl_sum.getSize()/2) / work_size;
	printf("compactify_sub2::nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("compactify_sub2::orig size: %d\n", cl_sum.getSize());
	if ((work_size*nb_blocks) != (cl_sum.getSize()/2)) {
		printf("compactify_sub2:: work_size not an even divider of nb_elem/2\n");
		exit(0);
		nb_blocks++;
	}

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;
	printf("compactify_sub2::global size: %d\n", global);
	printf("compactify_sub2::cl_sum size: %d\n", cl_sum.getSize());
	//exit(0);

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB2]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB2]->end();

	#if 1
	cl_sum.copyToHost();
	cl_sum_out.copyToHost();
	cl_sum_accu.copyToHost();
	int* in = cl_sum.getHostPtr();
	int* ou = cl_sum_out.getHostPtr();
	int* su = cl_sum_accu.getHostPtr();
	for (int i=0; i < cl_sum.getSize(); i++) {
		printf("cl_sum[%d], cl_sum_out= %d, %d\n", i, in[i], ou[i]);
	}
	for (int i=0; i < nb_blocks; i++) {
		printf("  cl_sum_accu[%d]= %d\n", i, su[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	//exit(0);
	#endif
}
//----------------------------------------------------------------------
void GE_SPH::sub2Sum(BufferGE<int>& cl_sum_accu, int work_size, int nb_blks, 
		             BufferGE<int>& cl_sum_accu_out)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			path = path + "/compactify_sub2sum_cl.cl";

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	sub2_sum_kernel = Kernel(ps->cli, strg, "sumScanSingleBlock");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	printf("inside sub2sum\n");

	int nb_blocks = 1;
	work_size = cl_sum_accu.getSize(); // 32 (16 * 128 = 2048)
	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size; // 16

	printf("compactify_sub2Sum::work_size= %d\n", work_size);
	printf("compactify_sub2Sum::global= %d\n", global);
	printf("compactify_sub2Sum::sub_blocks= %d\n", nb_blocks);
	printf("compactify_sub2Sum::sum_accu size= %d\n", cl_sum_accu.getSize());

	Kernel kern = sub2_sum_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_sum_accu_out.getDevicePtr()); 
	kern.setArg(iarg++, cl_sum_accu.getDevicePtr());
	kern.setArg(iarg++, cl_sum_accu.getSize()); // size; 2048
	#endif

	// would be much less if the arrays were compactified
	// nb blocks = nb grid cells
	

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB2SUM]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB2SUM]->end();

	// printout

	#if 1
	cl_sum_accu.copyToHost();
	cl_sum_accu_out.copyToHost();
	int* in = cl_sum_accu.getHostPtr();
	int* ou = cl_sum_accu_out.getHostPtr();
	for (int i=0; i < global; i++) {
		printf("cl_sum_accu,cl_sum_accu_out[%d]= %d, %d\n", i, in[i], ou[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	//exit(0);
	#endif
}
//----------------------------------------------------------------------
void GE_SPH::sub3(BufferGE<int>& cl_sum_out, int work_size, int nb_blks, BufferGE<int>& cl_sum_accu_out)
{
	#if 1
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			//path = path + "/block_scan_block64_cl.cl";
			path = path + "/compactify_sub3_cl.cl";

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_sub3_kernel = Kernel(ps->cli, strg, "compactifySub3Kernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	Kernel kern = compactify_sub3_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_sum_out.getDevicePtr());
	kern.setArg(iarg++, cl_sum_accu_out.getDevicePtr()); 
	kern.setArg(iarg++, cl_sum_out.getSize()); // size; 2048
	kern.setArg(iarg++, cl_sum_accu_out.getSize()); // size 16
	#endif

	// Each thread handles two elements of the block
	size_t nb_blocks = (cl_sum_out.getSize()) / work_size;

	nb_blocks = 16;
	work_size = 2048/nb_blocks;

	printf("compactify_sub3::nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("compactify_sub3::orig size: %d\n", cl_sum_out.getSize());
	if ((work_size*nb_blocks) != (cl_sum_out.getSize())) {
		printf("compactify_sub3:: work_size not an even divider of nb_elem/2\n");
		exit(0);
		nb_blocks++;
	}

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;
	printf("compactify_sub3::global size: %d\n", global);
	printf("compactify_sub3::cl_sum size: %d\n", cl_sum_out.getSize());
	//exit(0);

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB3]->start();

	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB3]->end();

	#if 1
	cl_sum_out.copyToHost();
	cl_sum_accu_out.copyToHost();
	int* in = cl_sum_out.getHostPtr();
	int* ou = cl_sum_accu_out.getHostPtr();
	for (int i=0; i < cl_sum_out.getSize(); i++) {
		printf("cl_sum_out[%d], %d\n", i, in[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);
	//exit(0);
	#endif
#endif
}
//----------------------------------------------------------------------
void GE_SPH::sub4(BufferGE<int>& cl_orig, int work_size, int nb_blks,  BufferGE<int>& cl_sum_out,
                  BufferGE<int>& cl_compact)
{
	static bool first_time = true;

	if (first_time) {
		try {
			string path(CL_SPH_UTIL_SOURCE_DIR);

			// Try blocks of size 64 (two warps of 32: need more shared mem)
			// efficiency. Still 32 threads per warp
			path = path + "/compactify_sub4_cl.cl";

			int length;
			char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	compactify_sub4_kernel = Kernel(ps->cli, strg, "compactifySub4Kernel");
			first_time = false;
		} catch(cl::Error er) {
			exit(1);
		}
	}

	Kernel kern = compactify_sub4_kernel;
	kern.setProfiling(true);

	int iarg = 0;
	#if 0
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	#else
	kern.setArg(iarg++, cl_compact.getDevicePtr());
	kern.setArg(iarg++, cl_orig.getDevicePtr());
	kern.setArg(iarg++, cl_sum_out.getDevicePtr()); 
	kern.setArg(iarg++, cl_compact.getSize()); // size  320k
	#endif

	printf("compactSub4::cl_compact_size: %d\n", cl_compact.getSize());

	// Each thread handles two elements of the block
	size_t nb_blocks = (cl_orig.getSize()) / work_size;

	//nb_blocks = 16;
	//work_size = 2048/nb_blocks;

	printf("compactify_sub4::nb_blocks= %d, work_size= %d\n", nb_blocks, work_size);
	printf("compactify_sub4::orig size: %d\n", cl_sum_out.getSize());
	if ((work_size*nb_blocks) != (cl_orig.getSize())) {
		printf("work_size*nb_blocks= %d\n", work_size*nb_blocks);
		printf("compactify_sub4:: work_size not an even divider of nb_elem/2\n");
		exit(0);
		nb_blocks++;
	}

#if 1
// FIRST 2048 are WRONG!!! WHY? 

	// global must be an integer multiple of work_size
	int global = nb_blocks * work_size;
	printf("compactify_sub4::global size: %d\n", global);
	//exit(0);

	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB4]->start();


	kern.execute(global, work_size);
	ps->cli->queue.finish();
	ts_cl[TI_COMPACTIFY_SUB4]->end();

	#if 1
	cl_orig.copyToHost();
	cl_compact.copyToHost();
	int* in = cl_orig.getHostPtr();
	int* ou = cl_compact.getHostPtr();
	for (int i=0; i < cl_compact.getSize(); i++) {
		printf("orig[%d]= %d, compact[%d], %d\n", i, in[i], i, ou[i]);
	}
	printf("global= %d\n", global);
	printf("work_size= %d, nb_blocks= %d\n", work_size, nb_blocks);

	cl_sum_out.copyToHost();
	int* cc = cl_sum_out.getHostPtr();

	for (int i=0; i < cl_sum_out.getSize(); i++) {
		printf(".. cl_sum_out[%d]= %d\n", i, cc[i]);
	}
	printf("sum_out size: %d\n", cl_sum_out.getSize());
	//exit(0);
	#endif
#endif
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

}; // namespace
