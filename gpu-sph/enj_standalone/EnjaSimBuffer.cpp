#include "EnjaSimBuffer.h"

namespace Enja
{
	//--------------------------------------------------
	//EnjaSimBuffer::EnjaSimBuffer(Enja::EnjaCudaHelper *enjaCudaHelper)
	EnjaSimBuffer::EnjaSimBuffer(EnjaCudaHelper *enjaCudaHelper)
		: //mParticlesMesh(particlesMesh)
		//, mEnjaVertexBuffer(NULL)
		  mEnjaCudaHelper(enjaCudaHelper)
		// **** ERROR ON NEXT LINE, CANNOT FIGURE IT OUT
		, SimBuffer(SimLib::BufferLocation::Device, sizeof(Enja::Vector4))
	{
	}

	//--------------------------------------------------
	EnjaSimBuffer::~EnjaSimBuffer()
	{
	}

	//--------------------------------------------------
	void EnjaSimBuffer::SetEnjaVertexBuffer(GLuint bufferid)
	{
		bool wasMapped = false;
		if(m_bufferid != bufferid)
		{
			if(mMapped)
			{
				wasMapped = true;
				UnmapBuffer();
			}

		}
		m_bufferid = bufferid;
		//mAllocedSize = mEnjaVertexBuffer->getSizeInBytes();
		//mSize = mAllocedSize;

		if(wasMapped)
			MapBuffer();

	}

	//--------------------------------------------------
	void EnjaSimBuffer::MapBuffer()
	{
		mEnjaCudaHelper->MapBuffer((void**)&mPtr, m_bufferid);
		mMapped = true;
	}

	//--------------------------------------------------
	void EnjaSimBuffer::UnmapBuffer()
	{
		mEnjaCudaHelper->UnmapBuffer((void**)&mPtr, m_bufferid);		
		mMapped = false;
	}

	//--------------------------------------------------
	size_t EnjaSimBuffer::GetSize()
	{
        //TODO: need to store our size or pass it in
		//return mEnjaVertexBuffer->getSizeInBytes();
	}

	//--------------------------------------------------
	void EnjaSimBuffer::Alloc(size_t size)
	{
		if(size == mAllocedSize)
			return;
		
		//mParticlesMesh->Resize(size/sizeof(float4));
	}

	//--------------------------------------------------
	void EnjaSimBuffer::Memset(int val)
	{
		//TODO
	}

	//--------------------------------------------------
	void EnjaSimBuffer::Free()
	{
		//TODO
	}
}
