#ifndef __OgreSimBuffer_h__
#define __OgreSimBuffer_h__

#include "Ogre.h"
#include "SimBuffer.h"
#include "OgreCudaHelper.h"
#include "OgreSimRenderable.h"

namespace SnowSim
{
	class OgreSimBuffer : public SimLib::SimBuffer
	{
	public:
		OgreSimBuffer::OgreSimBuffer(SnowSim::OgreSimRenderable *particlesMesh, SnowSim::OgreCudaHelper *OgreCudaHelper);
		~OgreSimBuffer();

		void SetOgreVertexBuffer(Ogre::HardwareVertexBufferSharedPtr  ogreVertexBuffer);

		virtual void MapBuffer();
		virtual void UnmapBuffer();

		virtual void Alloc(size_t size);
		virtual void Memset(int val);
		virtual void Free();
		virtual size_t GetSize();

	private:
		SnowSim::OgreSimRenderable *mParticlesMesh;
		SnowSim::OgreCudaHelper *mOgreCudaHelper;
		Ogre::HardwareVertexBufferSharedPtr mOgreVertexBuffer;
	};
	
}

#endif