
#include <stdio.h>
#include <memory.h>
#include <cutil.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>


typedef unsigned int uint;

texture<float, 1, cudaReadModeElementType> a_tex;
texture<float, 1, cudaReadModeElementType> b_tex;

__global__ void testKernel(float* a, float* b)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index > 100) index-=100;

	a[index] = 10;

	__threadfence();

	a[index] = tex1Dfetch(a_tex,index) + 1;

	volatile float x = tex1Dfetch(a_tex,index);
	x = tex1Dfetch(a_tex,index);
	x = tex1Dfetch(a_tex,index);
	x = tex1Dfetch(a_tex,index);


	__threadfence();
	__threadfence_block();
//	__threadfence_system();
	__syncthreads();

	//volatile float bv = a[index+1];
	b[index] = a[index+1];
	//b[index] = tex1Dfetch(a_tex,index+1);
}

void testKernel()
{
	float* da;
	float* db;
	cudaMalloc((void**)&da, 100*sizeof(float));
	cudaMalloc((void**)&db, 100*sizeof(float));
	cudaMemset(da,0,100*sizeof(float));
	cudaMemset(db,0,100*sizeof(float));

	float* ha;
	float* hb;
	cudaMallocHost((void**)&ha, 100*sizeof(float));
	cudaMallocHost((void**)&hb, 100*sizeof(float));
	memset(ha,0,100*sizeof(float));
	memset(hb,0,100*sizeof(float));


	cudaBindTexture(0, a_tex, da, 100*sizeof(float));
	cudaBindTexture(0, b_tex, db, 100*sizeof(float));

	testKernel<<<1,101>>>(da,db);

	cudaUnbindTexture(a_tex);
	cudaUnbindTexture(b_tex);

	cudaMemcpy(ha,da,100*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(hb,db,100*sizeof(float),cudaMemcpyDeviceToHost);

	return;
}


