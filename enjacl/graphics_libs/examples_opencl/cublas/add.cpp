#include <stdlib.h>
#include <cl.h>
#include <string>

using namespace std;

#include <array_opencl_1d.h>

// Assume a single kernel per program for simplicity

//----------------------------------------------------------------------
int main()
{
	bool profile = true;
	CL cl(profile); // can create cl wherever I wish

	string path(CL_SOURCE_DIR);
	path += "add.cl";

	cl_kernel kernel = cl.addProgram(path.c_str());

	int nx = 2*128;
	int ny = 2*128;
	int nz = 2*128;

	ArrayOpenCL1D<float> aa(nx,ny,nz);
	ArrayOpenCL1D<float> bb(nx,ny,nz);
	ArrayOpenCL1D<float> cc(nx,ny,nz);

	aa.setTo(1.);
	bb.setTo(2.);

	aa.copyToDevice();
	bb.copyToDevice();

	int ntot = nx*ny*nz;
	int buffer_size = ntot*sizeof(float);
	int count = ntot;

	int err;

	err = 0;

	// simplify interface 

	cl.setKernelArg(aa.getDevicePtr(), 0);
	cl.setKernelArg(bb.getDevicePtr(), 1);
	cl.setKernelArg(cc.getDevicePtr(), 2);
	cl.setKernelArg(count, 3); 

	size_t global = (size_t) count;
	size_t local = cl.getMaxWorkSize(kernel);

    cl_event exec = cl.execKernel(1, &global, &local);
	cl.waitForKernelsToFinish();

	cl.profile(); 
	cl.profile(exec);
	
	printf("===========================\n");
    exec = cl.execKernel(1, &global, &local);
	cl.waitForKernelsToFinish();
	cl.profile();

    // Read back the results from the device to verify the output
	float* results = new float [nx*ny*nz];

	cc.copyToHost();

	for (int i=0; i < 5; i++) {
		printf("res= %f\n", cc(i));
	}

	return(0);
}
//----------------------------------------------------------------------
