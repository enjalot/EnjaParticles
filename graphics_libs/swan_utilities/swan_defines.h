//#include "swan_stub.h"

#define dim3 block_config_t

#define cudaLoadProgramFromSource swanLoadProgramFromSource

//typedef long swanThread_t;

#define cudaInit swanInit

#define cudaInitThread swanInitThread
#define cudaWaitForThreads swanWaitForThreads

#define cudaGetDeviceCount swanGetDeviceCount

#define cudaSetDeviceNumber swanSetDeviceNumber
#define cudaSetTargetDevice swanSetTargetDevice

#define cudaSynchronize swanSynchronize
//#define cudaMalloc swanMalloc
#define cudaMallocPitch swanMallocPitch
#define cudaMemset swanMemset
#define cudaMallocHost swanMallocHost
#define cudaFree swanFree
#define cudaFreeHost swanFreeHost
#define cudaMemcpyHtoD swanMemcpyHtoD
#define cudaMemcpyDtoH swanMemcpyDtoH
#define cudaMemcpyDtoD swanMemcpyDtoD
#define cudaGetNumberOfComputeElements swanGetNumberOfComputeElements
#define cudaEnumerateDevices swanEnumerateDevices
#define cudaDeviceName swanDeviceName
#define cudaDeviceVersion swanDeviceVersion 
#define cudaDecompose swanDecompose
#define cudaMaxThreadCount swanMaxThreadCount

//===========================
#define __align__(x)
