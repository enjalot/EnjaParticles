#ifndef __CudaMax_cuh__
#define __CudaMax_cuh__

#include "cudpp/cudpp.h"

namespace SimLib
{
	class CudaMax
	{
	public:
		CudaMax(size_t elements);
		~CudaMax();
		
		float FindMax(float* d_idata);
	private:

		CUDPPHandle scanPlan;
		size_t mMemSize;
		size_t mElements;

		float* d_odata; 
		float* h_idata;
		float* h_odata;
	};

} // namespace SimLib

#endif