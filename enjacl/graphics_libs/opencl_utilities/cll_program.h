#ifndef _CL_PROGRAM_H_
#define _CL_PROGRAM_H_

#include <vector>
#include <string>

#include "cll_kernel.h"

// mac framework
#if defined (__APPLE_CC__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

class cll_Program
{
private:
    cl_program program;      // compute program
	std::string name;

public:
	std::vector<cll_Kernel> kernels;

public:
	cll_Program(cl_program _program);
	cll_Kernel  addKernel(const std::string kernel_name);
	cl_program  getProgram() {return program;}
	std::string getName() {return name;}
	void setName(std::string name) {this->name = name;}
};

#endif

