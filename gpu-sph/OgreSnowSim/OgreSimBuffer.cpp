#include "OgreSimBuffer.h"

namespace SnowSim
{
	OgreSimBuffer::OgreSimBuffer(SnowSim::OgreSimRenderable *particlesMesh, SnowSim::OgreCudaHelper *OgreCudaHelper)
		: mParticlesMesh(particlesMesh)
		, mOgreVertexBuffer(NULL)
		, mOgreCudaHelper(OgreCudaHelper)
		, SimBuffer(SimLib::BufferLocation::Device, sizeof(Ogre::Vector4))
	{
	}

	OgreSimBuffer::~OgreSimBuffer()
	{
	}

	void OgreSimBuffer::SetOgreVertexBuffer(Ogre::HardwareVertexBufferSharedPtr ogreVertexBuffer)
	{
		bool wasMapped = false;
		if(mOgreVertexBuffer != ogreVertexBuffer)
		{
			if(mMapped)
			{
				wasMapped = true;
				UnmapBuffer();
			}

		}
		mOgreVertexBuffer = ogreVertexBuffer;
		mAllocedSize = mOgreVertexBuffer->getSizeInBytes();
		mSize = mAllocedSize;

		if(wasMapped)
			MapBuffer();

	}

	void OgreSimBuffer::MapBuffer()
	{
		mOgreCudaHelper->MapBuffer((void**)&mPtr, mOgreVertexBuffer);
		mMapped = true;
	}

	void OgreSimBuffer::UnmapBuffer()
	{
		mOgreCudaHelper->UnmapBuffer((void**)&mPtr, mOgreVertexBuffer);		
		mMapped = false;
	}

	size_t OgreSimBuffer::GetSize()
	{
		return mOgreVertexBuffer->getSizeInBytes();
	}

	void OgreSimBuffer::Alloc(size_t size)
	{
		if(size == mAllocedSize)
			return;
		
		mParticlesMesh->Resize(size/sizeof(float4));
	}

	void OgreSimBuffer::Memset(int val)
	{
		//TODO
	}

	void OgreSimBuffer::Free()
	{
		//TODO
	}
}
