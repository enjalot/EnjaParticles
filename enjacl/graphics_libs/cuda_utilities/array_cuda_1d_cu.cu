#include <driver_types.h>

extern "C" 
void copyToDeviceFromHost_1d(void* dst, const void* src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}
//----------------------------------------------------------------------
extern "C"
void copyToHostFromDevice(void* dst, const void* src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}
//----------------------------------------------------------------------
extern "C" void cudaMalloc_1d(void** data , int nbBytes) 
{
	cudaError_t cudaError = cudaMalloc(data, nbBytes);
	// ERROR PROCESSING
}
//----------------------------------------------------------------------
extern "C" void clear_1d(void* data, size_t count)
{
	cudaError_t cudaError = cudaMemset(data, 0, count);
}
//----------------------------------------------------------------------
//extern "C" void cudaConfigureCall_ge(int gx, int gy, int gz, int bx, int by, int bz, size_t shared, int tokens)
// replace int by cudaStream_t (June 22, 2010)
extern "C" void cudaConfigureCall_ge(int gx, int gy, int gz, int bx, int by, int bz, size_t shared, cudaStream_t tokens)
{
	dim3 grid(gx,gy,gz);
	dim3 block(bx,by,bz);
	cudaConfigureCall(grid, block, shared, tokens);
}
//----------------------------------------------------------------------
extern "C" void cudaLaunch_ge(const char* entry)
{
	cudaLaunch(entry);
}
//----------------------------------------------------------------------
extern "C" void cudaSetupArgument_ge(void* arg, size_t count, size_t offset)
{
	cudaSetupArgument(arg, count, offset);
}
//----------------------------------------------------------------------
