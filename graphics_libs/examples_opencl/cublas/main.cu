#include <stdio.h>
#include <stdlib.h>
#include "cublas.h"

#include <array_cuda_1d.h>
#include <crt/host_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// defines CUDA_SAFE_CALL
#include <cutil.h>


//----------------------------------------------------------------------
void initialize_graphics()
{
	CUDA_SAFE_CALL(cudaSetDevice(0));
	//cublasInit();
}
//----------------------------------------------------------------------
void tstSaxpy()
{
	ArrayCuda1D<float> a(100,100,1);
	ArrayCuda1D<float> b(100,100,1);

	a.setTo(2.);
	b.setTo(1.);

	a.copyToDevice();
	b.copyToDevice();

	// b <- 2.*a + b
 	cublasSaxpy(a.getSize(), 2., a.getDevicePtr(), 1, b.getDevicePtr(), 1);
	b.copyToHost();

	for (int i=0; i < 10; i++) {
		printf("i,b= %d, %f\n", i, b(i));
	}
}
//----------------------------------------------------------------------
void tstSdot()
{
	ArrayCuda1D<float> a(100,100,1);
	ArrayCuda1D<float> b(100,100,1);

	a.setTo(1.);
	b.setTo(1.);

	a.copyToDevice();
	b.copyToDevice();

 	float dot = cublasSdot(a.getSize(), a.getDevicePtr(),1, b.getDevicePtr(), 1);
	printf("dot= %f\n", dot);
}
//----------------------------------------------------------------------
void tstSscal()
{
	ArrayCuda1D<float> a(100,100,1);
	ArrayCuda1D<float> b(100,100,1);

	a.setTo(.02);
	b.setTo(.03);

	a.copyToDevice();
	b.copyToDevice();

 	cublasSscal(a.getSize(), 3., a.getDevicePtr(),1);
	a.copyToHost();

	for (int i=0; i < 10; i++) {
		printf("a,b= %f, %f\n", a(i), b(i));
	}
}
//----------------------------------------------------------------------
int main()
{
	initialize_graphics();
	//tstSscal();
	//tstSdot();
	tstSaxpy();
	return 0;
}
//----------------------------------------------------------------------
