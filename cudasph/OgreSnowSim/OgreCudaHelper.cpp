
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "OgreCudaHelper.h"

using namespace Ogre;

namespace SnowSim
{
	OgreCudaHelper::OgreCudaHelper(SnowSim::Config *config, SimLib::SimCudaHelper *simCudaHelper)
	{
		mSnowConfig = config;
		mSimCudaHelper = simCudaHelper;
	}

	OgreCudaHelper::~OgreCudaHelper()
	{
	}

	void OgreCudaHelper::Initialize()
	{
		int cudaDevice = mSnowConfig->generalSettings.cudadevice;

		Ogre::RenderSystem* renderSystem = Ogre::Root::getSingleton().getRenderSystem();
		if(renderSystem->getName() == "Direct3D9 Rendering Subsystem")
		{
			D3D9RenderSystem* renderSystemD3D9 = static_cast<D3D9RenderSystem*>((void*)renderSystem);
			mRenderingMode = D3D9;
			// 		LPDIRECT3DDEVICE9 pD3DDevice;
			// 		mWindow->getCustomAttribute("D3DDEVICE",&pD3DDevice);

			IDirect3DDevice9 *pDxDevice = renderSystemD3D9->getActiveD3D9Device();
			//IDirect3DDevice9 *pDxDevice = D3D9RenderSystem::getActiveD3D9Device();

			mSimCudaHelper->InitializeD3D9(cudaDevice, pDxDevice);
		}
		else if(renderSystem->getName() == "OpenGL Rendering Subsystem")
		{
			GLRenderSystem* renderSystemGL = static_cast<GLRenderSystem*>((void*)renderSystem);
			mRenderingMode = GL;
			mSimCudaHelper->InitializeGL(cudaDevice);
		}
		else
		{
			mSimCudaHelper->Initialize(cudaDevice);
		}
	}
	void OgreCudaHelper::RegisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		switch(mRenderingMode)
		{
		case GL:
			{
				GLHardwareVertexBuffer* bufferGL = static_cast<GLHardwareVertexBuffer*>(hardwareBuffer.getPointer());
				GLuint bufferGL_ID = bufferGL->getGLBufferId();

				mSimCudaHelper->RegisterGLBuffer(bufferGL_ID);
			}
			break;
		case D3D9:
			{

				Ogre::D3D9HardwareVertexBuffer* bufferD3D9 = static_cast<Ogre::D3D9HardwareVertexBuffer*>(hardwareBuffer.getPointer());
				IDirect3DVertexBuffer9 *bufferD3D9_I = bufferD3D9->getD3D9VertexBuffer();

				mSimCudaHelper->RegisterD3D9Buffer(bufferD3D9_I);
			}
			break;
		default:
			assert(false);
		}
	}

	void OgreCudaHelper::UnregisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		switch(mRenderingMode)
		{
		case GL:
			{
				GLHardwareVertexBuffer* bufferGL = static_cast<GLHardwareVertexBuffer*>(hardwareBuffer.getPointer());
				GLuint bufferGL_ID = bufferGL->getGLBufferId();

				mSimCudaHelper->UnregisterGLBuffer(bufferGL_ID);				
			}
			break;
		case D3D9:
			{

				Ogre::D3D9HardwareVertexBuffer* bufferD3D9 = static_cast<Ogre::D3D9HardwareVertexBuffer*>(hardwareBuffer.getPointer());
				IDirect3DVertexBuffer9 *bufferD3D9_I = bufferD3D9->getD3D9VertexBuffer();

				mSimCudaHelper->UnregisterD3D9Buffer(bufferD3D9_I);
			}
			break;
		default:
			assert(false);
		}
	}
	
	void OgreCudaHelper::MapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		switch(mRenderingMode)
		{
		case GL:
			{
				GLHardwareVertexBuffer* bufferGL = static_cast<GLHardwareVertexBuffer*>(hardwareBuffer.getPointer());
				GLuint bufferGL_ID = bufferGL->getGLBufferId();

				mSimCudaHelper->MapBuffer(devPtr, bufferGL_ID);
			}
			break;
		case D3D9:
			{
				Ogre::D3D9HardwareVertexBuffer* bufferD3D9 = static_cast<Ogre::D3D9HardwareVertexBuffer*>(hardwareBuffer.getPointer());
				IDirect3DVertexBuffer9 *bufferD3D9_I = bufferD3D9->getD3D9VertexBuffer();

				mSimCudaHelper->MapBuffer(devPtr, bufferD3D9_I);
			}
			break;
		default:
			assert(false);
		}
	}

	void OgreCudaHelper::UnmapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		switch(mRenderingMode)
		{
		case GL:
			{
				GLHardwareVertexBuffer* bufferGL = static_cast<GLHardwareVertexBuffer*>(hardwareBuffer.getPointer());
				GLuint bufferGL_ID = bufferGL->getGLBufferId();

				mSimCudaHelper->UnmapBuffer(devPtr, bufferGL_ID);
			}
			break;
		case D3D9:
			{
				Ogre::D3D9HardwareVertexBuffer* bufferD3D9 = static_cast<Ogre::D3D9HardwareVertexBuffer*>(hardwareBuffer.getPointer());
				IDirect3DVertexBuffer9 *bufferD3D9_I = bufferD3D9->getD3D9VertexBuffer();

				mSimCudaHelper->UnmapBuffer(devPtr, bufferD3D9_I);
			}
		default:
			assert(false);
		}
	}
}
