#ifndef __SnowSimApp_h_
#define __SnowSimApp_h_

#include "SnowTerrain.h"
#include "SnowFluid.h"
#include "SnowGui.h"

#include "BaseApplication.h"
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#include "../res/resource.h"
#endif

namespace SnowSim
{
	class SnowApplication : public BaseApplication
	{
	public:
		SnowApplication(void);
		virtual ~SnowApplication(void);

	protected:
		void setupResources(void);

		void createScene(void);
		void destroyScene(void);

		bool frameRenderingQueued(const Ogre::FrameEvent& evt);
		bool frameStarted (const Ogre::FrameEvent &evt);
		bool frameEnded (const Ogre::FrameEvent &evt);

		bool keyPressed (const OIS::KeyEvent &e);
		void windowClosed(Ogre::RenderWindow* rw);

		Ogre::MaterialPtr buildDepthShadowMaterial(const Ogre::String& textureName);
		void changeShadows();
		void configureShadows(bool enabled, bool depthShadows);
		void createCamera(void);
		void setupControls();
		void createFrameListener();


		void itemSelected(OgreBites::SelectMenu* menu);
		void checkBoxToggled(OgreBites::CheckBox* box);

	private:

		bool mSimulationPaused;
		bool mDestroyed;

		SnowTerrain* mSnowTerrain;
		SnowFluid*	mSnowFluid;
		SnowGui*	mSnowGui;

		// Shadows...
		enum ShadowMode
		{
			SHADOWS_NONE = 0,
			SHADOWS_COLOUR = 1,
			SHADOWS_DEPTH = 2,
			SHADOWS_COUNT = 3
		};
		ShadowMode mShadowMode;

		OgreBites::SelectMenu* mShadowsMenu;
		//OgreBites::Label* mInfoLabel;
		Ogre::ShadowCameraSetupPtr mPSSMSetup;

		typedef std::list<Ogre::Entity*> EntityList;
		EntityList mHouseList;


		bool mScreenCapture;
		int mScreenCaptureFrame;
	};

}

#endif // #ifndef __SnowSim_h_