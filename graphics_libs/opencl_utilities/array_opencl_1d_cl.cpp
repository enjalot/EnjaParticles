//#include <driver_types.h>
#include "cl.h"

extern "C" 
void copyToDeviceFromHost_1d(cl_mem dst, const void* src, size_t count)
{
	cl_int status;
	CL cl;

	// count bytes
	// CL_TRUE: blocking
	// src: host
	cl.connectWriteBuffer(dst, CL_TRUE, count, src);
	//openclMemcpy(dst, src, count, openclMemcpyHostToDevice);
}
//----------------------------------------------------------------------
extern "C"
void copyToHostFromDevice(void* dst, const cl_mem src, size_t count)
{
	cl_int status;
	CL cl;

	// count bytes
	// CL_TRUE: blocking
	// src: device
	cl.connectReadBuffer(src, CL_TRUE, count, dst);
	//openclMemcpy(dst, src, count, openclMemcpyDeviceToHost);
}
//----------------------------------------------------------------------
// Should not be used outside this directory
extern "C" cl_mem openclMalloc_1d(int nbBytes) 
{
	CL cl; // should not be required.  (contex might change)
	cl_int status;

	if (nbBytes == 0) nbBytes = 1;

	cl_context ctx = CL::context; // static variable
	cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nbBytes, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("OpenCL could not allocate a R/W buffer of %d bytes\n", nbBytes);
		exit(0);
	}

	return out;
}
//----------------------------------------------------------------------
extern "C" void clear_1d(cl_mem data, size_t count)
{
	//FIX openclError_t openclError = openclMemset(data, 0, count);
}
//----------------------------------------------------------------------
extern "C" void openclConfigureCall_ge(int gx, int gy, int gz, int bx, int by, int bz, size_t shared, int tokens)
{
#if 0
	dim3 grid(gx,gy,gz);
	dim3 block(bx,by,bz);
	dim3 grid(gx,gy,gz);
	openclConfigureCall(grid, block, shared, tokens);
#endif
}
//----------------------------------------------------------------------
extern "C" void openclLaunch_ge(const char* entry)
{
	#if 0
	openclLaunch(entry);
	#endif
}
//----------------------------------------------------------------------
extern "C" void openclSetupArgument_ge(void* arg, size_t count, size_t offset)
{
	#if 0
	openclSetupArgument(arg, count, offset);
	#endif
}
//----------------------------------------------------------------------
