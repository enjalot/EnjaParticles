#ifndef __EnjaCudaHelper_h__
#define __EnjaCudaHelper_h__


#define SPHSIMLIB_3D_SUPPORT
//#include "SnowSimConfig.h"

//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
/*
#include "OgreHardwareBufferManager.h"
#include "OgreHardwareVertexBuffer.h"
#include "OgreGLRenderSystem.h"
#include "OgreGLHardwareVertexBuffer.h"
#include "OgreD3D9RenderSystem.h"
#include "OgreD3D9HardwareVertexBuffer.h"
*/

#include "SimCudaHelper.h"

namespace Enja
{

	class EnjaCudaHelper
	{
	public:
		//OgreCudaHelper::OgreCudaHelper(SnowSim::Config *config, SimLib::SimCudaHelper *simCudaHelper);
		EnjaCudaHelper(SimLib::SimCudaHelper *simCudaHelper);
		~EnjaCudaHelper();

		void Initialize();

		// CUDA REGISTER
		void RegisterHardwareBuffer(GLuint bufferid);
		void UnregisterHardwareBuffer(GLuint bufferid);

		// CUDA MAPPING
		void MapBuffer(void **devPtr, GLuint bufferid);
		void UnmapBuffer(void **devPtr, GLuint bufferid); 

	private:
		SimLib::SimCudaHelper *mSimCudaHelper;
	};
}


#endif
