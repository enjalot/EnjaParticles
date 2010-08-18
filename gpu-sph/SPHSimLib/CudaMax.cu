#include "CudaMax.cuh"

#include <cutil.h>

namespace SimLib
{

	CudaMax::CudaMax(size_t elements)
	{
		mElements = elements;
		mMemSize = elements*sizeof(float);

		// Scan configuration
		CUDPPConfiguration config;
		config.algorithm = CUDPP_SCAN;
		config.op = CUDPP_MAX;
		config.datatype = CUDPP_FLOAT;  
		config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
		cudppPlan(&scanPlan, config, elements, 1, 0);

		d_odata; CUDA_SAFE_CALL(cudaMalloc( (void**) &d_odata, mMemSize));
// 		h_idata = (float*)malloc(mMemSize); memset(h_idata,0, mMemSize);
// 		h_odata = (float*)malloc(mMemSize); memset(h_odata,0, mMemSize);
	}


	CudaMax::~CudaMax()
	{
		cudppDestroyPlan(scanPlan);
		CUDA_SAFE_CALL(cudaFree(d_odata));
// 		free(h_odata);
// 		free(h_idata);
	}

	float CudaMax::FindMax(float* d_idata)
	{
		cudppScan(scanPlan, d_odata, d_idata, mElements);

		float maxval;

		// just copy the max val, not the entire buffer
		CUDA_SAFE_CALL( cudaMemcpy( &maxval, d_odata, 1*sizeof(float), cudaMemcpyDeviceToHost) );


// 		CUDA_SAFE_CALL( cudaMemcpy( h_idata, d_idata, mMemSize, cudaMemcpyDeviceToHost) );
// 		CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mMemSize, cudaMemcpyDeviceToHost) );

		return maxval;
	}

} // namespace SimLib