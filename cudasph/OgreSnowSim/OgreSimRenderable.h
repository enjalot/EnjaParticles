#ifndef __OgreSimBufferMesh_h__
#define __OgreSimBufferMesh_h__

#include "Ogre.h"
#include "OgreCudaHelper.h"

typedef unsigned int uint;

namespace SnowSim
{
	class OgreSimBuffer;

	class OgreSimRenderable :  public Ogre::SimpleRenderable
	{
	public:
		OgreSimRenderable(OgreCudaHelper* ogreCudaHelper, uint numParticles);
		~OgreSimRenderable();

		void Resize(uint numParticles);

		OgreSimBuffer* GetCudaBufferPosition();
		OgreSimBuffer* GetCudaBufferColor();

	protected:
		void createMaterial();
		void fillHardwareBuffers();
		virtual Ogre::Real getBoundingRadius(void) const;
		virtual Ogre::Real getSquaredViewDepth(const Ogre::Camera *) const;    
		int mWidth;
		int mHeight;         

	private:
		OgreCudaHelper* mOgreCudaHelper;

		void createVertexBuffers();

		OgreSimBuffer *mCudaBufferPosition;
		OgreSimBuffer *mCudaBufferColor;

		Ogre::String mParticleMaterial;

		Ogre::HardwareVertexBufferSharedPtr mVertexBufferPosition;
		Ogre::HardwareVertexBufferSharedPtr mVertexBufferColor;

		bool mRegistered;
		int mVolumeSize;
		uint mNumParticles;
	};
}
#endif