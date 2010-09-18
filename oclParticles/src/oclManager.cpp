/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "particleSystem_common.h"
#include "particleSystem_engine.h"

////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;

extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeBitonicSort(void);

//Context initialization/deinitialization
extern "C" void startupOpenCL(int argc, const char **argv){
    cl_platform_id cpPlatform;
    cl_device_id cdDevice;
    cl_int ciErrNum;
    
    // Get the NVIDIA platform
    //shrLog("oclGetPlatformID...\n\n"); 
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get the devices
    //shrLog("clGetDeviceIDs...\n\n"); 
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Create the context
    //shrLog("clCreateContext...\n\n"); 
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // List used device 
    //shrLog("GPU Device being used:\n"); 
    oclPrintDevInfo(LOGBOTH, cdDevice);

    //Create a command-queue
    //shrLog("clCreateCommandQueue...\n\n"); 
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    initBitonicSort(cxGPUContext, cqCommandQueue, argv);
    initParticles(cxGPUContext, cqCommandQueue, argv);
}

extern "C" void shutdownOpenCL(void){
    cl_int ciErrNum;
    closeParticles();
    closeBitonicSort();
    ciErrNum  = clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//GPU buffer allocation
extern "C" void allocateArray(memHandle_t *memObj, size_t size){
    cl_int ciErrNum;
    //shrLog(" clCreateBuffer (GPU GMEM, %u bytes)...\n\n", size); 
    *memObj = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void freeArray(memHandle_t memObj){
    cl_int ciErrNum;
    ciErrNum = clReleaseMemObject(memObj);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//host<->device memcopies
extern "C" void copyArrayFromDevice(void *hostPtr, memHandle_t memObj, unsigned int vbo, size_t size){
    cl_int ciErrNum;
    assert( vbo == 0 );
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size){
    cl_int ciErrNum;
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

//Register/unregister OpenGL buffer object to/from Compute context
extern "C" void registerGLBufferObject(uint vbo){
}

extern "C" void unregisterGLBufferObject(uint vbo){
}

//Map/unmap OpenGL buffer object to/from Compute buffer
extern "C" memHandle_t mapGLBufferObject(uint vbo){
    return NULL;
}

extern "C" void unmapGLBufferObject(uint vbo){
}
