#include <stdlib.h>
#include <cl.h>
#include <string>

using namespace std;

#include <array_opencl_1d.h>

// Assume a single kernel per program for simplicity

//----------------------------------------------------------------------
int main()
{
	CL cl;
	printf("cl context: %d\n", cl.context);

	string path = "/Users/erlebach/grav/branch_grav/branch1/graphics_libs/examples_opencl/add_arrays/add.cl";
	cl_kernel kernel = cl.addProgram(path.c_str());

	int nx = 128;
	int ny = 128;
	int nz = 128;
	ArrayOpenCL1D<float> aa(nx,ny,nz);
	ArrayOpenCL1D<float> bb(nx,ny,nz);
	ArrayOpenCL1D<float> cc(nx,ny,nz);
	//CLArray1D a(nx,ny,nz);
	//CLArray1D b(nx,ny,nz);
	//CLArray1D c(nx,ny,nz);

	//a.setTo(1.);
	//b.setTo(2.);

	aa.setTo(1.);
	bb.setTo(2.);

	aa.copyToDevice(); // ERROR!!!
	bb.copyToDevice();

	int ntot = nx*ny*nz;
	float* a = new float [ntot];
	float* b = new float [ntot];
	float* res = new float [ntot];

	for (int i=0; i < ntot; i++) {
		a[i] = 1.;
		b[i] = 2.;
		res[i] = 0.;
	}

	int buffer_size = ntot*sizeof(float);
	int count = ntot;

	printf("sizeof(cl_mem)= %d\n", sizeof(cl_mem));
	cl_mem a_d = cl.createReadBuffer(buffer_size); // in bytes
	cl_mem b_d = cl.createReadBuffer(buffer_size); // in bytes
	cl_mem res_d = cl.createWriteBuffer(buffer_size); // in bytes
    cl_event a_event = cl.connectWriteBuffer(a_d, CL_TRUE, sizeof(float)*count,  a);
    cl_event b_event = cl.connectWriteBuffer(b_d, CL_TRUE, sizeof(float)*count, b);

	int err;

	err = 0;
	//err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
	// ERROR NEXT LINE!!
	cl_mem aad = aa.getDevicePtr();
	cl_mem bbd = bb.getDevicePtr();
	cl_mem ccd = cc.getDevicePtr();
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aad); //&aa.getDevicePtr());
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }   

	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bbd);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &ccd);

	//err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
	//err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_d);
printf("count bef: %d\n", count);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
printf("count aft: %d\n", count);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }   

	size_t global = (size_t) count;
	size_t local = cl.getMaxWorkSize(kernel);
printf("local= %d, global= %d\n", local, global);

    cl_event exec = cl.execKernel(kernel, 1, &global, &local);
	cl.waitForKernelsToFinish();

    // Read back the results from the device to verify the output
	float* results = new float [nx*ny*nz];
	//float results[nx*ny*nz]; // Why is this incorrect?
// ERROR ON NEXT LINE
	printf("count=  %d\n", count);
    cl_event read_event = cl.connectReadBuffer(res_d, CL_TRUE, sizeof(float)*count, results);

	cc.copyToHost();

	for (int i=0; i < 5; i++) {
		//printf("res= %f\n", results[i]);
		printf("res= %f\n", cc(i));
	}
exit(0);



	//add.exec(a,b,c);

	//c.copyToHost();
	//printf("c(3,3,3)= %f\n", c(3,3,3));
	return(0);
}
//----------------------------------------------------------------------
