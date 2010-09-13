//#include <driver_types.h>
#include <swan_api.h>

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
	printf("before swanMalloc\n");
	*data = (void*) swanMalloc((size_t) nbBytes);
	// ERROR PROCESSING
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
	dim3 grid(gx,gy,gz);
	dim3 block(bx,by,bz);
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
