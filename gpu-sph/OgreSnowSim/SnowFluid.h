#ifndef __SnowFluid_h_
#define __SnowFluid_h_

#include "SnowSimConfig.h"
#include "OgreCudaHelper.h"

#include <OgrePlatform.h>
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>

#include <OISEvents.h>
#include <OISInputManager.h>
#include <OISKeyboard.h>
#include <OISMouse.h>

#include <SdkTrays.h>
#include <SdkCameraMan.h>

#include "OgreTerrain.h"
#include "OgreTerrainGroup.h"
#include "OgreTerrainQuadTreeNode.h"
#include "OgreTerrainMaterialGeneratorA.h"
#include "OgreTerrainPaging.h"

#include "SnowTerrain.h"
#include "OgreCudaHelper.h"
#include "OgreSimBuffer.h"

#define SPHSIMLIB_3D_SUPPORT
#include "SimulationSystem.h"


namespace SnowSim
{
	class SnowFluid : public Ogre::FrameListener//, public OIS::KeyListener, public OIS::MouseListener
	{
	public:
		SnowFluid(SnowSim::Config *config);
		~SnowFluid(void);

		void createScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr, SnowTerrain* terrain, Ogre::Light* terrainLight);
		void destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr);

		bool frameRenderingQueued(const Ogre::FrameEvent& evt);
		bool frameStarted (const Ogre::FrameEvent &evt);
		bool frameEnded (const Ogre::FrameEvent &evt);

		bool keyPressed (OIS::Keyboard* keyboard, const OIS::KeyEvent &e);

		Ogre::SceneNode* mParticlesNode;
		OgreSimRenderable *mParticlesEntity;

	protected:

	private:
		Ogre::RenderWindow* mRenderWindow;
		void SnowFluid::setParticleMaterial(Ogre::String particleMaterial);
		void configureTerrain(SnowTerrain* terrain);
		
		void SetScene(int scene);
		void FillTestData(int scene, SimLib::Sim::ParticleData &hParticles) ;

		SnowSim::Config *mSnowConfig;
		SnowSim::OgreCudaHelper* mOgreCudaHelper;
		SimLib::SimCudaHelper* mSimCudaHelper;

		int lastScene;
		bool mProgress;

		SimLib::SimulationSystem* mParticleSystem;
		int mNumParticles;
		int mVolumeSize;

		Ogre::Entity* sphereEntity;
		Ogre::SceneNode* sphereNode;

		Ogre::Vector3 spherePosition;
		Ogre::Vector3 sphereVelocity;
		Ogre::Vector3 sphereAccel;


		Ogre::ManualObject* mFluidGridObject;
		Ogre::SceneNode* mFluidGridNode;

	};
}
#endif // #ifndef __SnowFluid_h_