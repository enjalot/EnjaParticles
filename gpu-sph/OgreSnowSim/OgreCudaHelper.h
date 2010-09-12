#ifndef __OgreCudaHelper_h__
#define __OgreCudaHelper_h__


#define SPHSIMLIB_3D_SUPPORT
#include "SnowSimConfig.h"

//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>

#include "OgreHardwareBufferManager.h"
#include "OgreHardwareVertexBuffer.h"
#include "OgreGLRenderSystem.h"
#include "OgreGLHardwareVertexBuffer.h"
#include "OgreD3D9RenderSystem.h"
#include "OgreD3D9HardwareVertexBuffer.h"

#include "SimCudaHelper.h"

namespace SnowSim
{

	class OgreCudaHelper
	{
	public:
		OgreCudaHelper::OgreCudaHelper(SnowSim::Config *config, SimLib::SimCudaHelper *simCudaHelper);
		~OgreCudaHelper();

		void Initialize();

		// CUDA REGISTER
		void RegisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer);
		void UnregisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer);

		// CUDA MAPPING
		void MapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr bufObj);
		void UnmapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr bufObj); 

	private:
		SnowSim::Config *mSnowConfig;
		SimLib::SimCudaHelper *mSimCudaHelper;

		enum RenderingMode
		{
			GL = 0,
			D3D9 = 1,
		};
		RenderingMode mRenderingMode;

	};
}


#endif
