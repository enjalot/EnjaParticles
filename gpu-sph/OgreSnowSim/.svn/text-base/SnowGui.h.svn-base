#ifndef __SnowGui_h_
#define __SnowGui_h_

#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>

#include "SnowTerrain.h"
#include "SnowFluid.h"
#include "SnowSimConfig.h"

#include "MyGUI.h"
#include "MyGUI_OgrePlatform.h"
#include "BaseManager.h"

namespace SnowSim
{
	class SnowGui : public base::BaseManager
	{
	public:;
	   SnowGui(SnowSim::Config *config);
	   ~SnowGui(void);

	   void createScene(MyGUI::Gui *mGUI, Ogre::SceneManager* mSceneMgr, SnowSim::SnowTerrain* terrain, SnowSim::SnowFluid* fluid);
	   void destroyScene(MyGUI::Gui *mGUI, Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr);

	   void notifyChangePanels(int _key, size_t _value);
	   virtual void setupResources();

	private:
		SnowSim::Config *mSnowConfig;


// 		PanelViewWindow* mView;
// 		PanelDirector* mPanelDirector;
// 		PanelDynamic* mPanelDynamic;
// 		PanelStatic* mPanelStatic;

	};
}

#endif 