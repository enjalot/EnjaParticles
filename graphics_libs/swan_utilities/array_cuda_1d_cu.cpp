//#include <driver_types.h>
#include <swan_api.h>
#include "swan_defines.h"
#include <stdlib.h>

extern "C" 
void copyToDeviceFromHost_1d(void* dst, const void* src, size_t count)
{
	//cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	swanMemcpyHtoD((void*) src, dst, (size_t) count);
}
//----------------------------------------------------------------------
extern "C"
void copyToHostFromDevice(void* dst, const void* src, size_t count)
{
	//cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	swanMemcpyDtoH((void*) src, dst, (size_t) count);
}
//----------------------------------------------------------------------
extern "C" void cudaMalloc_1d(void** data , int nbBytes) 
{
	//cudaError_t cudaError = cudaMalloc(data, nbBytes);
	//printf("cudaMalloc_1d::before swanMalloc\n");
	*data = (void*) swanMalloc((size_t) nbBytes);
	//printf("cudaMalloc_1d, sizeof(void*) = %d\n", sizeof(void*));
	// ERROR PROCESSING
	//printf("remove EXIT\n"); exit(0);
}
//----------------------------------------------------------------------
extern "C" void clear_1d(void* data, size_t count)
{
	//cudaError_t cudaError = cudaMemset(data, 0, count);
	swanMemset(data, 0, (size_t) count);
}
//----------------------------------------------------------------------
extern "C" void cudaConfigureCall_ge(int gx, int gy, int gz, int bx, int by, int bz, size_t shared, int tokens)
{
	dim3 grid;
	grid.x = gx;
	grid.y = gy;
	grid.z = gz;
	dim3 block;
	block.x = bx; 
	block.y = by; 
	block.z = bz; 
	// If I need this, I'll have problems
	//GE cudaConfigureCall(grid, block, shared, tokens);
}
//----------------------------------------------------------------------
extern "C" void cudaLaunch_ge(const char* entry)
{
	// If I need this, I'll have problems
	//GE cudaLaunch(entry);
}
//----------------------------------------------------------------------
extern "C" void cudaSetupArgument_ge(void* arg, size_t count, size_t offset)
{
	// If I need this, I'll have problems
	//GE cudaSetupArgument(arg, count, offset);
}
//----------------------------------------------------------------------
