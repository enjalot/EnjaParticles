#ifndef _CL_KERNEL_H_
#define _CL_KERNEL_H_

#include <stdlib.h>
#include <vector>
#include <string>

// mac framework
#if defined (__APPLE_CC__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

class cll_Kernel
{
private:
    cl_kernel kernel;   // compute kernel
	std::string name;

public:
	cll_Kernel(std::string name);
	template <class T> int setArg(T arg, int which_arg);
	cl_event exec(cl_uint work_dim, const size_t *global_work_size, const size_t *local_work_size);
	cl_kernel getKernel() {return kernel;}
	void setKernel(cl_kernel kernel) {this->kernel = kernel;}
	std::string getName() {return name;}
	void setName(std::string name) {this->name = name;}
};

//----------------------------------------------------------------------
template <class T>
int cll_Kernel::setArg(T arg, int which_arg)
{
	int err = clSetKernelArg(kernel, which_arg, sizeof(T), &arg); 

    if (err != CL_SUCCESS) {
        printf("Error: CL::cll_Kernel::setKernelArg,  Failed to set kernel arguments! %d\n", err);
		printf("sizeof(argument)= %d\n", sizeof(T));
        exit(1);
    }   
	return err;
}
//----------------------------------------------------------------------

#endif
