#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
//#include "opengl.h"

#include "CudaHelpers.h"

#include <cuda_gl_interop.h>

extern "C" 
{




int PrintDevices(int deviceCount, int deviceSelected)
{
    cudaError_t err = cudaSuccess;

    cudaDeviceProp deviceProperty;
    for (int currentDeviceId = 0; currentDeviceId < deviceCount; ++currentDeviceId)
    {
        memset(&deviceProperty, 0, sizeof(cudaDeviceProp));
        err = cudaGetDeviceProperties(&deviceProperty, currentDeviceId);

        printf("\ndevice name: %s", deviceProperty.name);
        if (currentDeviceId == deviceSelected)
        {
            printf("    <----- creating CUcontext on this");    
        }
        printf("\n");

        printf("device sharedMemPerBlock: %d \n", deviceProperty.sharedMemPerBlock);
        printf("device totalGlobalMem: %d \n", deviceProperty.totalGlobalMem);
        printf("device regsPerBlock: %d \n", deviceProperty.regsPerBlock);
        printf("device warpSize: %d \n", deviceProperty.warpSize);
        printf("device memPitch: %d \n", deviceProperty.memPitch);
        printf("device maxThreadsPerBlock: %d \n", deviceProperty.maxThreadsPerBlock);
        printf("device maxThreadsDim[0]: %d \n", deviceProperty.maxThreadsDim[0]);
        printf("device maxThreadsDim[1]: %d \n", deviceProperty.maxThreadsDim[1]);
        printf("device maxThreadsDim[2]: %d \n", deviceProperty.maxThreadsDim[2]);
        printf("device maxGridSize[0]: %d \n", deviceProperty.maxGridSize[0]);
        printf("device maxGridSize[1]: %d \n", deviceProperty.maxGridSize[1]);
        printf("device maxGridSize[2]: %d \n", deviceProperty.maxGridSize[2]);
        printf("device totalConstMem: %d \n", deviceProperty.totalConstMem);
        printf("device major: %d \n", deviceProperty.major);
        printf("device minor: %d \n", deviceProperty.minor);
        printf("device clockRate: %d \n", deviceProperty.clockRate);
        printf("device textureAlignment: %d \n", deviceProperty.textureAlignment);
        printf("device deviceOverlap: %d \n", deviceProperty.deviceOverlap);
        printf("device multiProcessorCount: %d \n", deviceProperty.multiProcessorCount);

        printf("\n");
    }

    return err;
}

void CUDA_Init(int dev) 
{
	cudaError res;
	int count;
	cudaDeviceProp p;
	
    res = cudaGetDeviceCount(&count);
	if(res != cudaSuccess)
	{
		CUDA_CheckError("cudaGetDeviceCount failed");
	}
	
	//LOG("CUDA_Init: %d available devices \n", count);

    PrintDevices(count, dev);

    if(dev >= count || dev < 0)
	{
		dev = cutGetMaxGflopsDeviceId();		
	}
    
	//LOG("CUDA_Init: Using device %d\n", dev);
	
	res = cudaSetDevice(dev);
	if(res != cudaSuccess)
	{
		CUDA_CheckError("cudaSetDevice failed");
	}
	else 
	{		
		//LOG("CUDA_Init: Successfull cudaSetDevice\n", dev);
	}

	res = cudaGLSetGLDevice(dev);
	if(res != cudaSuccess)
	{
		CUDA_CheckError("cudaGLSetGLDevice failed");
	}
	else 
	{		
		//LOG("CUDA_Init: Successfull cudaGLSetGLDevice\n", dev);
	}
}

cudaError_t CUDA_GLMapBufferObject(void **devPtr, GLuint bufObj)
{
	return cudaGLMapBufferObject(devPtr, bufObj);
}

cudaError_t CUDA_GLUnmapBufferObject(void **devPtr, GLuint bufObj)
{
	cudaError err = cudaGLUnmapBufferObject(bufObj);
	*devPtr = 0;
	return err;
}

// cudaError_t CUDA_CreateVBO(GLuint &vbo, uint size) 
// {
// 	vbo = CreateVBO(size);
// 	return CUDA_RegisterVBO(vbo);
// }
// 
// cudaError_t CUDA_CreateDataVBO(GLuint &vbo, uint size, GLvoid *data) 
// {
// 	vbo = CreateVBO(size, data);
// 	return CUDA_RegisterVBO(vbo);
// }
// 
// cudaError_t CUDA_FreeVBO(GLuint &vbo) 
// {
// 	cudaError err = CUDA_UnregisterVBO(vbo);
// 	FreeVBO(vbo);
// 	return err;
// }

cudaError_t CUDA_RegisterVBO(GLuint vbo) 
{
    return cudaGLRegisterBufferObject(vbo);
}

cudaError_t CUDA_UnregisterVBO(GLuint vbo) 
{
    return cudaGLUnregisterBufferObject(vbo);
}

void CUDA_CheckError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
//		DIE("Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		/*
		FILE *fp = fopen("error.txt", "w");
        fprintf(fp, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		fclose(fp);
        exit(EXIT_FAILURE);
		*/
    }                         
}




} // extern "C"

