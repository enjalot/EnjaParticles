#ifndef __EnjaSimBuffer_h__
#define __EnjaSimBuffer_h__

#include "SimBuffer.h"
#include "EnjaCudaHelper.h"
//#include "EnjaSimRenderable.h"

namespace Enja
{
	class EnjaSimBuffer : public SimLib::SimBuffer
	{
	public:
		EnjaSimBuffer::EnjaSimBuffer(SnowSim::EnjaCudaHelper *EnjaCudaHelper);
		~EnjaSimBuffer();

		void SetEnjaVertexBuffer(GLuint  bufferid);

		virtual void MapBuffer();
		virtual void UnmapBuffer();

		virtual void Alloc(size_t size);
		virtual void Memset(int val);
		virtual void Free();
		virtual size_t GetSize();

	private:
		//SnowSim::EnjaSimRenderable *mParticlesMesh;
		SnowSim::EnjaCudaHelper *mEnjaCudaHelper;
        GLuint m_bufferid;
	};
	
}

#endif
