#include <stdio.h>
#include "cll_program.h"
#include "cll_kernel.h"

#include <timege.h>
extern GE::Time clock_dbg("dbg", -1, 3);

//----------------------------------------------------------------------
cll_Program::cll_Program(cl_program _program)
{
	this->program = _program;
}
//----------------------------------------------------------------------
cll_Kernel cll_Program::addKernel(const std::string kernel_name)
{
	//clock_dbg.begin();

	cll_Kernel ck(kernel_name);

	int err = 0;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);

	if (err != 0) {
		printf("addKernel: error= %d\n", err);
	}

	ck.setKernel(kernel);

	//clock_dbg.end();

	return ck;
}
//----------------------------------------------------------------------
