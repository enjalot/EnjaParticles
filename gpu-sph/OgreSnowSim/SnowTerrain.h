#ifndef __SnowTerrain_h_
#define __SnowTerrain_h_

#include "SnowSimConfig.h"
#include "OgreCudaHelper.h"

#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>

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

#include "OgreHardwareVertexBuffer.h"

namespace SnowSim
{

	class SnowTerrain : public Ogre::FrameListener//, public OIS::KeyListener, public OIS::MouseListener
	{
	public:
		SnowTerrain(SnowSim::Config *config);
		~SnowTerrain(void);

		void createScene(Ogre::SceneManager* mSceneMgr, Ogre::Light* terrainLight);
		void destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr);

		Ogre::TerrainMaterialGeneratorA::SM2Profile* getMaterialProfile();

		bool frameRenderingQueued(const Ogre::FrameEvent& evt);
		bool keyPressed (const OIS::KeyEvent &e);

		void SaveTerrains(bool onlyIfModified);
		void dumpTextures();

		Ogre::Terrain* getTerrain();

		float* getTerrainHeightData();
		Ogre::Vector4* getTerrainNormalData();
		int getTerrainSize();
		Ogre::Real getTerrainWorldSize();

		Ogre::Vector3 mTerrainPos;
		Ogre::Terrain* mTerrain;
		Ogre::TerrainGroup* mTerrainGroup;

	protected:
		void defineTerrain(long x, long y, bool flat = false);
		void initBlendMaps(Ogre::Terrain* terrain);
		
		Ogre::Terrain* createTerrain();

	private:
		Ogre::Vector4* convertNormalsToFloats(Ogre::PixelBox* terrainNormals, bool compressed);
		Ogre::ManualObject* createDebugNormals(Ogre::SceneManager* mSceneMgr);

		SnowSim::Config *mSnowConfig;

		Ogre::uint mTerrainSize;
		Ogre::Real mTerrainWorldSize;
		Ogre::Real mTerrainWorldScale;

		//Terrain stuff
		Ogre::uint8 mLayerEdit;
		Ogre::Real mUpdateCountDown;
		Ogre::Real mUpdateRate;

		bool mTerrainsImported;

		Ogre::ManualObject* mDebugNormalsManualObject;
		Ogre::SceneNode* mDebugNormalsNode;
		Ogre::SceneManager* mSceneMgr;
		Ogre::TerrainGlobalOptions* mTerrainGlobals;

	};

}
#endif // #ifndef __SnowTerrain_h_