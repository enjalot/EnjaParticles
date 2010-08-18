#include "SnowGui.h"


using namespace Ogre;

namespace SnowSim
{
	SnowGui::SnowGui(SnowSim::Config *snowConfig)
		: mSnowConfig(snowConfig)
// 		, mView(nullptr)
// 		, mPanelDirector(nullptr)
// 		, mPanelDynamic(nullptr)
// 		, mPanelStatic(nullptr)
	{
	}

	SnowGui::~SnowGui()
	{
	}
	void SnowGui::setupResources()
	{
//  		base::BaseManager::setupResources();
//  		addResourceLocation(getRootMedia() + "/Demos/Demo_PanelView");
//  		addResourceLocation(getRootMedia() + "/Common/Wallpapers");
	}

	void SnowGui::createScene(MyGUI::Gui *mGUI, Ogre::SceneManager* mSceneMgr, SnowSim::SnowTerrain* terrain, SnowSim::SnowFluid* fluid)
	{
 		//MyGUI::LanguageManager::getInstance().loadUserTags("core_theme_black_blue_tag.xml");
 		//mGUI->load("core_skin.xml");

 		// quick mygui test
//   		MyGUI::ButtonPtr button = mGUI->createWidget<MyGUI::Button>("Button", 10, 10, 300, 26, MyGUI::Align::Default, "Main");
//   		button->setCaption("exit");

// 		mGUI->load("Wallpaper0.layout");
// 		MyGUI::VectorWidgetPtr& root = MyGUI::LayoutManager::getInstance().load("BackHelp.layout");
// 		root.at(0)->findWidget("Text")->setCaption("Panel View control implementation.");

// 		mView = new PanelViewWindow();
// 		mPanelDirector = new PanelDirector();
// 		mPanelDynamic = new PanelDynamic();
// 		mPanelStatic = new PanelStatic();
// 
// 		mPanelDirector->eventChangePanels = MyGUI::newDelegate(this, &DemoKeeper::notifyChangePanels);
// 		mView->getPanelView()->addItem(mPanelDirector);
// 		mView->getPanelView()->addItem(mPanelDynamic);
// 		mView->getPanelView()->addItem(mPanelStatic);
	}

	void SnowGui::destroyScene(MyGUI::Gui *mGUI, Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr)
	{
// 		mView->getPanelView()->removeAllItems();
// 
// 		delete mView;
// 		mView = nullptr;
// 		delete mPanelDirector;
// 		mPanelDirector = nullptr;
// 		delete mPanelDynamic;
// 		mPanelDynamic = nullptr;
// 		delete mPanelStatic;
// 		mPanelStatic = nullptr;
	}

	void SnowGui::notifyChangePanels(int _key, size_t _value)
	{
// 		if (_key == EVENT_SHOW_STATIC)
// 		{
// 			mView->getPanelView()->setItemShow(mPanelStatic, _value != 0);
// 		}
// 		else if (_key == EVENT_SHOW_DYNAMIC)
// 		{
// 			mView->getPanelView()->setItemShow(mPanelDynamic, _value != 0);
// 		}
// 		else if (_key == EVENT_COUNT_DYNAMIC)
// 		{
// 			mPanelDynamic->setVisibleCount(_value);
// 		}
	}

}