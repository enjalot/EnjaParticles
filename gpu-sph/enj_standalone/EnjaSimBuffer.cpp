#include "EnjaSimBuffer.h"

namespace Enja
{
	//--------------------------------------------------
	//EnjaSimBuffer::EnjaSimBuffer(Enja::EnjaCudaHelper *enjaCudaHelper)
	EnjaSimBuffer::EnjaSimBuffer(EnjaCudaHelper *enjaCudaHelper)
		: //mParticlesMesh(particlesMesh)
		//, mEnjaVertexBuffer(NULL)
		  mEnjaCudaHelper(enjaCudaHelper)
		, SimBuffer(SimLib::Device, sizeof(Vec4))
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
        printf("m_bufferid: %d\n", m_bufferid);
        printf("bufferid: %d\n", bufferid);
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

        printf("wasMapped: %d\n", wasMapped);
		if(wasMapped)
			MapBuffer();

	}

	//--------------------------------------------------
	void EnjaSimBuffer::MapBuffer()
	{
		mEnjaCudaHelper->MapBuffer((void**)&mPtr, m_bufferid);
        printf("Map buffer: %d\n", mPtr);
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
