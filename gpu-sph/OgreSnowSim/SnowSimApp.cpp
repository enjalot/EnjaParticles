#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "SnowSimApp.h"

using namespace Ogre;
using namespace OgreBites;

namespace SnowSim 
{

	//-------------------------------------------------------------------------------------
	SnowApplication::SnowApplication(void) 
		: mSimulationPaused(false)
		, mScreenCaptureFrame(0)
		, mScreenCapture(false)
	{
		mSnowGui = new SnowGui(mSnowConfig);
	}

	//-------------------------------------------------------------------------------------
	SnowApplication::~SnowApplication(void)
	{
		delete mSnowGui;
	}

	void SnowApplication::setupResources(void)
	{
		BaseApplication::setupResources();

		mSnowGui->setupResources();
	}

	//-------------------------------------------------------------------------------------
	void SnowApplication::destroyScene(void)
	{
		if(mDestroyed) return;

		mDestroyed = true;

		// Destroy GUI
		mSnowGui->destroyScene(mGUI, mWindow, mSceneMgr);

		// Destroy fluid
		mSnowFluid->destroyScene(mWindow, mSceneMgr);

		// Destroy terrain
		mSnowTerrain->destroyScene(mWindow, mSceneMgr);

		delete mSnowTerrain;
		delete mSnowFluid;
	}

	//-------------------------------------------------------------------------------------
	void SnowApplication::createScene(void)
	{	
		mDestroyed = false;

		srand ( time(NULL) );

		mSnowTerrain = new SnowTerrain(mSnowConfig);
		mSnowFluid = new SnowFluid(mSnowConfig);

 		MaterialManager::getSingleton().setDefaultTextureFiltering(TFO_ANISOTROPIC);
 		MaterialManager::getSingleton().setDefaultAnisotropy(7);

		// Setup lighting
 		//Vector3 lightdir(0.55, -0.3, 0.75);
 		Vector3 lightdir(00, -0.85, 0);
 		lightdir.normalise();
	 
	 	// PRIMARY LIGHT
 		Light* primaryLight = mSceneMgr->createLight("PrimaryLight");
 		primaryLight->setType(Light::LT_DIRECTIONAL);
 		primaryLight->setDirection(lightdir);
 		primaryLight->setDiffuseColour(ColourValue::White);
 		primaryLight->setSpecularColour(ColourValue(0.4, 0.4, 0.4));
	 
 		mSceneMgr->setAmbientLight(ColourValue(0.2, 0.2, 0.2));

 		mWindow->getViewport(0)->setBackgroundColour(mSnowConfig->sceneSettings.backgroundColor);
		//ColourValue fadeColour(0.9, 0.9, 0.9);
// 		mSceneMgr->setFog(FOG_LINEAR, fadeColour, 0.0, 500, 5000);
		//mSceneMgr->setFog(FOG_EXP, fadeColour, 0.005);
		//mSceneMgr->setFog(FOG_EXP2, fadeColour, 0.003);

		// Set a skybox
		mSceneMgr->setSkyBox(mSnowConfig->sceneSettings.skyBoxMaterial.length()>0, mSnowConfig->sceneSettings.skyBoxMaterial);

		// Create terrain
		if(mSnowConfig->terrainSettings.enabled)
  			mSnowTerrain->createScene(mSceneMgr, primaryLight);

		// Create fluid
		mSnowFluid->createScene(mWindow, mSceneMgr, mSnowTerrain, primaryLight);
		//mCamera->lookAt(mSnowFluid->mParticlesNode->getPosition());;

		// Create GUI
		mSnowGui->createScene(mGUI, mSceneMgr, mSnowTerrain, mSnowFluid);

		// Place camera 
		Vector3 cameraPos =mSnowConfig->sceneSettings.cameraPosition;
		if(mSnowConfig->sceneSettings.cameraRelativeToFluid && mSnowConfig->fluidSettings.enabled)
			cameraPos += mSnowFluid->mParticlesNode->getPosition();

		mCamera->setPosition(cameraPos);
		//mCamera->setDirection(0,100,0);
		mCamera->setOrientation(mSnowConfig->sceneSettings.cameraOrientation);
	// 	mCamera->setPosition(mSnowTerrain->mTerrainPos + Vector3(1683, 50, 2116));
	// 	mCamera->lookAt(mSnowTerrain->mTerrainPos);
	// 	mCamera->setNearClipDistance(5);
	// 	mCamera->setNearClipDistance(0.5);
	// 	mCamera->setFarClipDistance(500);
		mCamera->setNearClipDistance(0.1);
		mCamera->setFarClipDistance(50000);
		if (mRoot->getRenderSystem()->getCapabilities()->hasCapability(RSC_INFINITE_FAR_PLANE))
		{
			mCamera->setFarClipDistance(0);   // enable infinite far clip distance if we can
		}

		return;
	}

	void SnowApplication::createFrameListener()
	{
		BaseApplication::createFrameListener();

		setupControls();
	}

	bool SnowApplication::frameStarted (const FrameEvent &evt)
	{
		if(mDestroyed) return false;

		mSnowFluid->frameStarted(evt);
		return BaseApplication::frameStarted(evt);
	}

	bool SnowApplication::frameEnded (const FrameEvent &evt)
	{
		if(mDestroyed) return false;

		mSnowFluid->frameEnded(evt);
		return BaseApplication::frameEnded(evt);

	}

	bool SnowApplication::frameRenderingQueued(const Ogre::FrameEvent& evt)
	{
		if(mDestroyed) return false;

		if(mScreenCapture)
		{
			char tmp[1000]={0};
			sprintf(tmp,"scr%i.png",mScreenCaptureFrame++);
			mWindow->writeContentsToFile(tmp);
		}

 		if(!mSimulationPaused) {
			mSnowTerrain->frameRenderingQueued(evt);
			mSnowFluid->frameRenderingQueued(evt);
		}

		return BaseApplication::frameRenderingQueued(evt);  // don't forget the parent updates!
	}


	bool SnowApplication::keyPressed (const OIS::KeyEvent &evt)
	{
		// toggle visibility of help dialog
		if (evt.key == OIS::KC_H || evt.key == OIS::KC_F1)   
		{
			if (!mTrayMgr->isDialogVisible())  mTrayMgr->showOkDialog("Help", "");
			else mTrayMgr->closeDialog();
		}

		// don't process any more keys if dialog is up
		if (mTrayMgr->isDialogVisible()) return true;   


		switch (evt.key)
		{
		case OIS::KC_SYSRQ:
			// take a screenshot
			{
				mWindow->writeContentsToTimestampedFile("screenshot", ".png");
			}
			break;
		case OIS::KC_F:   
			{
				mTrayMgr->toggleAdvancedFrameStats();
			}
			break;
		case OIS::KC_R:
			// cycle polygon rendering mode
			{
				Ogre::String newVal;
				Ogre::PolygonMode pm;

				switch (mCamera->getPolygonMode())
				{
				case Ogre::PM_SOLID:
					newVal = "Wireframe";
					pm = Ogre::PM_WIREFRAME;
					break;
				case Ogre::PM_WIREFRAME:
					newVal = "Points";
					pm = Ogre::PM_POINTS;
					break;
				default:
					newVal = "Solid";
					pm = Ogre::PM_SOLID;
				}

				mCamera->setPolygonMode(pm);
//				mDetailsPanel->setParamValue(10, newVal);
			}
			break;
		case OIS::KC_F5:
			{
				// refresh all textures
				Ogre::TextureManager::getSingleton().reloadAll();
			}
			break;

		case OIS::KC_S:
			// CTRL-S to save
			if (mKeyboard->isKeyDown(OIS::KC_LCONTROL) || mKeyboard->isKeyDown(OIS::KC_RCONTROL))
			{
				Ogre::LogManager::getSingleton().logMessage(LogMessageLevel::LML_CRITICAL, "Saving terrain");
				mSnowTerrain->SaveTerrains(false);
			}
			break;		
		case OIS::KC_F9:
			mScreenCapture = !mScreenCapture;
			if(mScreenCapture)
				mScreenCaptureFrame = 0;
			break;
		case OIS::KC_F10:
			// dump
			{
				mSnowTerrain->dumpTextures();
			}
		case OIS::KC_F11:
			// dump
			{
				mSnowTerrain->getTerrain();
			}
		case OIS::KC_P:
			// dump
			{
				mSimulationPaused = !mSimulationPaused;
			}
			break;
		}


		mSnowTerrain->keyPressed(evt);

		mSnowFluid->keyPressed(mKeyboard, evt);

		return BaseApplication::keyPressed(evt);
	}


	/*-----------------------------------------------------------------------------
	| Extends setupView to change some initial camera settings for this sample.
	-----------------------------------------------------------------------------*/
	void SnowApplication::createCamera()
	{
		BaseApplication::createCamera();

	}


	void SnowApplication::setupControls()
	{
		mCameraMan->setTopSpeed(500);

		setDragLook(true);

		mTrayMgr->showCursor();

		// make room for the controls
		//mTrayMgr->showLogo(TL_TOPRIGHT);
		mTrayMgr->showFrameStats(TL_TOPRIGHT);
		mTrayMgr->toggleAdvancedFrameStats();

//		mInfoLabel = mTrayMgr->createLabel(TL_TOP, "TInfo", "", 350);

		// 	mEditMenu = mTrayMgr->createLongSelectMenu(TL_BOTTOM, "EditMode", "Edit Mode", 370, 250, 3);
		// 	mEditMenu->addItem("None");
		// 	mEditMenu->addItem("Elevation");
		// 	mEditMenu->addItem("Blend");
		// 	mEditMenu->selectItem(0);  // no edit mode

		// 	mFlyBox = mTrayMgr->createCheckBox(TL_BOTTOM, "Fly", "Fly");
		// 	mFlyBox->setChecked(false, false);

	// 	mShadowsMenu = mTrayMgr->createLongSelectMenu(TL_BOTTOM, "Shadows", "Shadows", 370, 250, 3);
	// 	mShadowsMenu->addItem("None");
	// 	mShadowsMenu->addItem("Colour Shadows");
	// 	mShadowsMenu->addItem("Depth Shadows");
	// 	mShadowsMenu->selectItem(0);  // no edit mode

		// a friendly reminder
	// 	StringVector names;
	// 	names.push_back("Help");
	// 	mTrayMgr->createParamsPanel(TL_TOPLEFT, "Help", 100, names)->setParamValue(0, "H/F1");
	}


	void SnowApplication::itemSelected(SelectMenu* menu)
	{
	// 	if (menu == mEditMenu)
	// 	{
	// 		mMode = (Mode)mEditMenu->getSelectionIndex();
	// 	}
	// 	else 
			if (menu == mShadowsMenu)
		{
			mShadowMode = (ShadowMode)mShadowsMenu->getSelectionIndex();
			changeShadows();
		}
	}

	void SnowApplication::checkBoxToggled(CheckBox* box)
	{
	// 	if (box == mFlyBox)
	// 	{
	// 		mFly = mFlyBox->isChecked();
	// 	}
	}

	void SnowApplication::windowClosed(Ogre::RenderWindow* rw)
	{

		BaseApplication::windowClosed(rw);
	}


	MaterialPtr SnowApplication::buildDepthShadowMaterial(const String& textureName)
	{
		String matName = "DepthShadows/" + textureName;

		MaterialPtr ret = MaterialManager::getSingleton().getByName(matName);
		if (ret.isNull())
		{
			MaterialPtr baseMat = MaterialManager::getSingleton().getByName("Ogre/shadow/depth/integrated/pssm");
			ret = baseMat->clone(matName);
			Pass* p = ret->getTechnique(0)->getPass(0);
			p->getTextureUnitState("diffuse")->setTextureName(textureName);

			Vector4 splitPoints;
			const PSSMShadowCameraSetup::SplitPointList& splitPointList = 
				static_cast<PSSMShadowCameraSetup*>(mPSSMSetup.get())->getSplitPoints();
			for (int i = 0; i < 3; ++i)
			{
				splitPoints[i] = splitPointList[i];
			}
			p->getFragmentProgramParameters()->setNamedConstant("pssmSplitPoints", splitPoints);


		}

		return ret;
	}

	void SnowApplication::changeShadows()
	{
		configureShadows(mShadowMode != SHADOWS_NONE, mShadowMode == SHADOWS_DEPTH);
	}

	void SnowApplication::configureShadows(bool enabled, bool depthShadows)
	{
		TerrainMaterialGeneratorA::SM2Profile* matProfile = mSnowTerrain->getMaterialProfile();

		matProfile->setReceiveDynamicShadowsEnabled(enabled);
	#ifdef SHADOWS_IN_LOW_LOD_MATERIAL
		matProfile->setReceiveDynamicShadowsLowLod(true);
	#else
		matProfile->setReceiveDynamicShadowsLowLod(false);
	#endif

		// Default materials
		for (EntityList::iterator i = mHouseList.begin(); i != mHouseList.end(); ++i)
		{
			(*i)->setMaterialName("Examples/TudorHouse");
		}

		if (enabled)
		{
			// General scene setup
			mSceneMgr->setShadowTechnique(SHADOWTYPE_TEXTURE_ADDITIVE_INTEGRATED);
			mSceneMgr->setShadowFarDistance(3000);

			// 3 textures per directional light (PSSM)
			mSceneMgr->setShadowTextureCountPerLightType(Ogre::Light::LT_DIRECTIONAL, 3);

			if (mPSSMSetup.isNull())
			{
				// shadow camera setup
				PSSMShadowCameraSetup* pssmSetup = new PSSMShadowCameraSetup();
				pssmSetup->setSplitPadding(mCamera->getNearClipDistance());
				pssmSetup->calculateSplitPoints(3, mCamera->getNearClipDistance(), mSceneMgr->getShadowFarDistance());
				pssmSetup->setOptimalAdjustFactor(0, 2);
				pssmSetup->setOptimalAdjustFactor(1, 1);
				pssmSetup->setOptimalAdjustFactor(2, 0.5);

				mPSSMSetup.bind(pssmSetup);

			}
			mSceneMgr->setShadowCameraSetup(mPSSMSetup);

			if (depthShadows)
			{
				mSceneMgr->setShadowTextureCount(3);
				mSceneMgr->setShadowTextureConfig(0, 2048, 2048, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureConfig(1, 1024, 1024, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureConfig(2, 1024, 1024, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureSelfShadow(true);
				mSceneMgr->setShadowCasterRenderBackFaces(true);
				mSceneMgr->setShadowTextureCasterMaterial("PSSM/shadow_caster");

				MaterialPtr houseMat = buildDepthShadowMaterial("fw12b.jpg");
				for (EntityList::iterator i = mHouseList.begin(); i != mHouseList.end(); ++i)
				{
					(*i)->setMaterial(houseMat);
				}

			}
			else
			{
				mSceneMgr->setShadowTextureCount(3);
				mSceneMgr->setShadowTextureConfig(0, 2048, 2048, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureConfig(1, 1024, 1024, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureConfig(2, 1024, 1024, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureSelfShadow(false);
				mSceneMgr->setShadowCasterRenderBackFaces(false);
				mSceneMgr->setShadowTextureCasterMaterial(StringUtil::BLANK);
			}

			matProfile->setReceiveDynamicShadowsDepth(depthShadows);
			matProfile->setReceiveDynamicShadowsPSSM(static_cast<PSSMShadowCameraSetup*>(mPSSMSetup.get()));

		}
		else
		{
			mSceneMgr->setShadowTechnique(SHADOWTYPE_NONE);
		}


	}


}