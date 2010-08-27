
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "EnjaCudaHelper.h"


namespace Enja
{
	//--------------------------------------------------
	EnjaCudaHelper::EnjaCudaHelper(SimLib::SimCudaHelper *simCudaHelper)
	{
		mSimCudaHelper = simCudaHelper;
	}

	//--------------------------------------------------
	EnjaCudaHelper::~EnjaCudaHelper()
	{
	}

	//--------------------------------------------------
	void EnjaCudaHelper::Initialize()
	{
		//int cudaDevice = mSnowConfig->generalSettings.cudadevice;
        //need to get cudaDevice some other way (look at console test?)

		// HARDCODED: BAD!!!
		int cudaDevice = 1;
		mSimCudaHelper->InitializeGL(cudaDevice);
	}

	//--------------------------------------------------
	void EnjaCudaHelper::RegisterHardwareBuffer(GLuint bufferid)
	{
		mSimCudaHelper->RegisterGLBuffer(bufferid);
	}

	//--------------------------------------------------
	void EnjaCudaHelper::UnregisterHardwareBuffer(GLuint bufferid)
	{
		mSimCudaHelper->UnregisterGLBuffer(bufferid);
	}
	
	//--------------------------------------------------
	void EnjaCudaHelper::MapBuffer(void **devPtr, GLuint bufferid)
	{
	    mSimCudaHelper->MapBuffer(devPtr, bufferid);
	}

	//--------------------------------------------------
	void EnjaCudaHelper::UnmapBuffer(void **devPtr, GLuint bufferid)
	{
		mSimCudaHelper->UnmapBuffer(devPtr, bufferid);
	}
}
