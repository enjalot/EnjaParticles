#include "OgreSimRenderable.h"
#include "OgreSimBuffer.h"

namespace SnowSim
{
	OgreSimRenderable::OgreSimRenderable(OgreCudaHelper *ogreCudaHelper, uint numParticles)
			: mOgreCudaHelper(ogreCudaHelper)
			, mNumParticles(0)
			, mVolumeSize(1024)
			, mRegistered(FALSE)
	{

		mWidth  = mVolumeSize;
		mHeight = mVolumeSize;

		mCudaBufferPosition = new OgreSimBuffer(this, mOgreCudaHelper);
		mCudaBufferColor = new OgreSimBuffer(this, mOgreCudaHelper);

		// set bounding box
		//mBox = Ogre::AxisAlignedBox( Ogre::Vector3(0, 0, 0), Ogre::Vector3(mVolumeSize, mVolumeSize, mVolumeSize) );
		mBox.setInfinite();
		
		Resize(numParticles);

		Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create("CudaVertexBufferMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		material->createTechnique()->createPass();
		material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
		material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
		material->getTechnique(0)->getPass(0)->setVertexColourTracking(Ogre::TVC_DIFFUSE);
		setMaterial("CudaVertexBufferMaterial");
	}

	OgreSimRenderable::~OgreSimRenderable()
	{
		//TODO: free shit
	}

	void OgreSimRenderable::Resize(uint numParticles)
	{
		if(numParticles == mNumParticles)
			return;

		mNumParticles = numParticles;
		if(mRegistered)
		{
			mOgreCudaHelper->UnregisterHardwareBuffer(mVertexBufferPosition);
			mOgreCudaHelper->UnregisterHardwareBuffer(mVertexBufferColor);
			mRegistered = false;
		}

		assert(!mCudaBufferColor->IsMapped());
		assert(!mCudaBufferPosition->IsMapped());

		createVertexBuffers();

		mOgreCudaHelper->RegisterHardwareBuffer(mVertexBufferPosition);
		mOgreCudaHelper->RegisterHardwareBuffer(mVertexBufferColor);
		mRegistered = true;

		mCudaBufferPosition->SetOgreVertexBuffer(mVertexBufferPosition);
		mCudaBufferColor->SetOgreVertexBuffer(mVertexBufferColor);
	}

	OgreSimBuffer* OgreSimRenderable::GetCudaBufferPosition()
	{
		return mCudaBufferPosition;
	}

	OgreSimBuffer* OgreSimRenderable::GetCudaBufferColor()
	{
		return mCudaBufferColor;
	}


	Ogre::Real OgreSimRenderable::getBoundingRadius(void) const
	{
		return 0;
	}


	Ogre::Real OgreSimRenderable::getSquaredViewDepth(const Ogre::Camera *) const
	{
		return 0;
	}

	void OgreSimRenderable::createVertexBuffers()
	{
		//particlesSubMesh->useSharedVertices = false;

		// our vertexes are just points
		mRenderOp.operationType = Ogre::RenderOperation::OT_POINT_LIST;
		mRenderOp.useIndexes = false;
		mRenderOp.vertexData = new Ogre::VertexData();

		mRenderOp.vertexData->vertexCount = mNumParticles;

		mRenderOp.vertexData->vertexBufferBinding->unsetAllBindings();

		// POSITIONS

		// define the vertex format
		size_t currOffset = 0;
		mRenderOp.vertexData->vertexDeclaration->addElement(0, currOffset, Ogre::VET_FLOAT4, Ogre::VES_POSITION);
		currOffset += Ogre::VertexElement::getTypeSize(Ogre::VET_FLOAT4);
		
		// allocate the vertex buffer
		mVertexBufferPosition = Ogre::HardwareBufferManager::getSingleton().createVertexBuffer(
			mRenderOp.vertexData->vertexDeclaration->getVertexSize(0),
			mRenderOp.vertexData->vertexCount, 
			Ogre::HardwareBuffer::HBU_DISCARDABLE,
			false);		

		// bind positions to 0	
		mRenderOp.vertexData->vertexBufferBinding->setBinding(0, mVertexBufferPosition);


		// COLORS

		// define the vertex format
		currOffset = 0;
		mRenderOp.vertexData->vertexDeclaration->addElement(1, currOffset, Ogre::VET_FLOAT4, Ogre::VES_DIFFUSE);
		currOffset += Ogre::VertexElement::getTypeSize(Ogre::VET_FLOAT4);

		// allocate the color buffer
		mVertexBufferColor = Ogre::HardwareBufferManager::getSingleton().createVertexBuffer(
			mRenderOp.vertexData->vertexDeclaration->getVertexSize(0),
			mRenderOp.vertexData->vertexCount, 
			Ogre::HardwareBuffer::HBU_DISCARDABLE,
			false);		

		// bind colors to 1
		mRenderOp.vertexData->vertexBufferBinding->setBinding(1, mVertexBufferColor);

		// Fill some random positions for particles
		Ogre::Vector4* pVertexPos = static_cast<Ogre::Vector4*>(mVertexBufferPosition->lock(Ogre::HardwareBuffer::HBL_NORMAL));
		for(uint i=0;i<mNumParticles;i++)
		{
			Ogre::Vector3 pos = Ogre::Vector3(rand()%mVolumeSize,rand()%mVolumeSize,rand()%mVolumeSize);
			Ogre::Vector4 p = Ogre::Vector4(pos);
			pVertexPos[i] = p;
		}
		mVertexBufferPosition->unlock();

		// Fill some colors for particles
		Ogre::RGBA* pVertexColor = static_cast<Ogre::RGBA*>(mVertexBufferColor->lock(Ogre::HardwareBuffer::HBL_NORMAL));
		Ogre::RenderSystem* rs = Ogre::Root::getSingleton().getRenderSystem();
		for(uint i=0;i<mNumParticles;i++)
		{
			rs->convertColourValue(Ogre::ColourValue::Red, &pVertexColor[i]);
		}
		mVertexBufferColor->unlock();

	}
}