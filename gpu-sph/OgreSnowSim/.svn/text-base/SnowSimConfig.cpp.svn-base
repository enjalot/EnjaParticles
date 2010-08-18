#include "SnowSimConfig.h"

using namespace Ogre;

namespace SnowSim {

	Config::Config(const std::string& configFileName)
		: mCfg(NULL)
	{
		Ogre::LogManager::getSingleton().logMessage("*** Loading OgreSnow configuration ***");

		//mCfg->load(configFileName);
		mCfg = new Ogre::ConfigFile();
		mCfg->loadFromResourceSystem(configFileName, "Essential");

		loadConfig();
	}

	Config::~Config()
	{
	}

	void Config::loadConfig()
	{
		// defaults
		generalSettings.cudadevice = 0;
		generalSettings.logLevel = LoggingLevel::LL_LOW;
		generalSettings.showOgreConfigDialog = true;
		generalSettings.showOgreGui = true;

		Ogre::ConfigFile::SettingsIterator iter = mCfg->getSettingsIterator("General");
		while(iter.hasMoreElements())
		{
			String name = iter.peekNextKey();
			Ogre::StringUtil::toLowerCase(name);
			String value = iter.getNext();

			if(name == "cudadevice")
			{
				generalSettings.cudadevice = StringConverter::parseUnsignedInt(value);
			}
			else if(name == "loglevel")
			{
				generalSettings.logLevel = (Ogre::LoggingLevel)StringConverter::parseUnsignedInt(value);
			}
			else if(name == "showogreconfigdialog")
			{
				generalSettings.showOgreConfigDialog = (Ogre::LoggingLevel)StringConverter::parseBool(value);
			}
			else if(name == "showogregui")
			{
				generalSettings.showOgreGui = (Ogre::LoggingLevel)StringConverter::parseBool(value);
			}
			else if(name == "fluidshader")
			{
				generalSettings.fluidShader = value;
			}
		}

		loadSceneConfig();
		loadFluidConfig();
		loadTerrainConfig();
	}

	void Config::loadSceneConfig()
	{
		// defaults
		//sceneSettings.skyBoxMaterial = "Examples/CloudyNoonSkyBox";

		sceneSettings.fluidScene = 9;

		Ogre::ConfigFile::SettingsIterator iter = mCfg->getSettingsIterator("Scene");
		while(iter.hasMoreElements())
		{
			String name = iter.peekNextKey();
			Ogre::StringUtil::toLowerCase(name);
			String value = iter.getNext();
			
			if(name == "skyboxmaterial")
			{
				sceneSettings.skyBoxMaterial = value;
			}
			else if(name == "cameratelativetofluid")
			{
				sceneSettings.cameraRelativeToFluid = StringConverter::parseBool(value);
			}
			else if(name == "cameraposition")
			{
				sceneSettings.cameraPosition = StringConverter::parseVector3(value);
			}	
			else if(name == "cameraorientation")
			{
				sceneSettings.cameraOrientation = StringConverter::parseQuaternion(value);
			}
			else if(name == "fluidposition")
			{
				sceneSettings.fluidPosition = StringConverter::parseVector3(value);
			}
			else if(name == "terrainposition")
			{
				sceneSettings.terrainPosition = StringConverter::parseVector3(value);
			}
			else if(name == "backgroundcolor")
			{
				sceneSettings.backgroundColor = StringConverter::parseColourValue(value);
			}
			else if(name == "fluidgridcolor")
			{
				sceneSettings.fluidGridColor = StringConverter::parseColourValue(value);
			}
			else if(name == "fluidscene")
			{
				sceneSettings.fluidScene = StringConverter::parseUnsignedInt(value);
			}
		}
	}

	void Config::loadFluidConfig()
	{
		// defaults
		fluidSettings.simpleSPH = true;
		fluidSettings.enabled = false;
		fluidSettings.showFluidGrid = true;

		Ogre::ConfigFile::SettingsIterator iter = mCfg->getSettingsIterator("Fluid");
		while(iter.hasMoreElements())
		{
			String name = iter.peekNextKey();
			Ogre::StringUtil::toLowerCase(name);
			String value = iter.getNext();
			
			if(name == "simplesph")
			{
				fluidSettings.simpleSPH = StringConverter::parseBool(value);
			}
			if(name == "enabled")
			{
				fluidSettings.enabled = StringConverter::parseBool(value);
			}
			if(name == "enablekerneltiming")
			{
				fluidSettings.enableKernelTiming = StringConverter::parseBool(value);
			}
			else if(name == "showfluidgrid")
			{
				fluidSettings.showFluidGrid = StringConverter::parseBool(value);
			}
			else if(name == "gridwallcollisions")
			{
				fluidSettings.gridWallCollisions = StringConverter::parseBool(value);
			}
			else if(name == "terraincollisions")
			{
				fluidSettings.terrainCollisions = StringConverter::parseBool(value);
			}
		}
	}

	void Config::loadTerrainConfig()
	{
		Ogre::ConfigFile::SettingsIterator iter = mCfg->getSettingsIterator("Terrain");

		// defaults
		terrainSettings.enabled = false;
		terrainSettings.showDebugNormals = false;
		terrainSettings.flat = true;
		//terrainSettings.heightDataFile = "terrain_2048_alpine3_height_raw32.raw";
		//terrainSettings.normalsDataFile = "terrain_2048_alpine3_normal.bmp";
		terrainSettings.worldSize = 2250.0f;
		terrainSettings.worldScale = 376.0f;
		terrainSettings.size = 4097;

/*
		normalheightImages[1] = "terrain_1024_alpine3_shader_white_normalheight.png";
		normalheightImages[2] = "terrain_1024_alpine3_shader_white_normalheight.png";
		normalheightImages[3] = "terrain_1024_alpine3_shader_white_normalheight.png";

		blendImages[0] = "";
		blendImages[1] = "terrain_4096_alpine3_select_thinflowsdeep0.bmp";
		blendImages[2] = "terrain_4096_alpine3_select_sediment0_sedimente.bmp";
		blendImages[3] = "terrain_4096_alpine3_select_selection6.bmp";
*/


		iter = mCfg->getSettingsIterator("Terrain");
		while(iter.hasMoreElements())
		{
			String name = iter.peekNextKey();
			Ogre::StringUtil::toLowerCase(name);
			String value = iter.getNext();

			if(name == "enabled")
			{
				terrainSettings.enabled = StringConverter::parseBool(value);
			}
			else if(name == "showdebugnormals")
			{
				terrainSettings.showDebugNormals = StringConverter::parseBool(value);
			}
			else if(name == "flat")
			{
				terrainSettings.flat = StringConverter::parseBool(value);
			}
			else if(name == "size")
			{
				terrainSettings.size = StringConverter::parseUnsignedInt(value);
			}
			else if(name == "worldsize")
			{
				terrainSettings.worldSize = StringConverter::parseReal(value);
			}
			else if(name == "worldscale")
			{
				terrainSettings.worldScale = StringConverter::parseReal(value);
			}
			else if(name == "normalsdatafile")
			{
				terrainSettings.normalsDataFile = value;
			}
			else if(name == "heightdatafile")
			{
				terrainSettings.heightDataFile = value;
			}
		}

		Ogre::String setting;

		for(int i = 0; i< 10; i++)
		{
			setting = "textureBlendFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureBlendFile = mCfg->getSetting(setting, "Terrain");
			terrainSettings.textureBlendFileList.push_back(textureBlendFile);

			setting = "textureLayerNormalHeightFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureLayerNormalHeightFile = mCfg->getSetting(setting, "Terrain");

			setting = "textureLayerDiffSpecFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureLayerDiffSpecFile = mCfg->getSetting(setting, "Terrain");
			
			setting = "textureLayerSize"+Ogre::StringConverter::toString(i);
			Ogre::Real textureLayerSize = StringConverter::parseReal(mCfg->getSetting(setting, "Terrain"));

			if(textureLayerNormalHeightFile.length() > 0 || textureLayerDiffSpecFile.length() > 0 || textureLayerSize > 0)
			{
				terrainSettings.textureLayerNormalHeightFileList.resize(i+1);
				terrainSettings.textureLayerDiffSpecFileList.resize(i+1);
				terrainSettings.textureLayerSize.resize(i+1);

				if(textureLayerSize == 0)
					textureLayerSize = terrainSettings.worldSize;

				terrainSettings.textureLayerNormalHeightFileList[i] = textureLayerNormalHeightFile;
				terrainSettings.textureLayerDiffSpecFileList[i] = textureLayerDiffSpecFile;
				terrainSettings.textureLayerSize[i] = textureLayerSize;
			}

		}
	}
}