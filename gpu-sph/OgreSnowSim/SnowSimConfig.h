#ifndef __SnowSim_h_
#define __SnowSim_h_

#include <Ogre.h>
#include "SimulationSystem.h"

namespace SnowSim
{
	struct GeneralSettings
	{
		int cudadevice;
		Ogre::LoggingLevel logLevel;
		bool showOgreConfigDialog;
		bool showOgreGui;
		Ogre::String fluidShader;
	};

	struct SceneSettings
	{
		Ogre::String skyBoxMaterial;
		bool cameraRelativeToFluid;
		Ogre::Vector3 cameraPosition;
		Ogre::Quaternion cameraOrientation;

		Ogre::Vector3 fluidPosition;
		Ogre::Vector3 terrainPosition;

		Ogre::ColourValue backgroundColor;
		Ogre::ColourValue fluidGridColor;

		int fluidScene;
	};

	struct FluidSettings
	{
		bool simpleSPH;
		bool enabled;
		bool enableKernelTiming;
		bool showFluidGrid;
		bool gridWallCollisions;
		bool terrainCollisions;
	};

	
	typedef std::vector<Ogre::String> TextureLayerFileList;
	typedef std::vector<Ogre::Real> TextureLayerSizeList;

	struct TerrainSettings
	{
		bool enabled;

		int size;

		// horizontal scale
		Ogre::Real worldSize;
		// vertical scale
		Ogre::Real worldScale;

		Ogre::String normalsDataFile;
		Ogre::String heightDataFile;

		// blend files
		TextureLayerFileList textureBlendFileList;

		// texture layer files
		TextureLayerFileList textureLayerDiffSpecFileList;
		TextureLayerFileList textureLayerNormalHeightFileList;
		TextureLayerSizeList textureLayerSize;
		bool flat;
		bool showDebugNormals;
	};

	class Config
	{
	public:
		Config(const std::string& configFileName = "SnowSim.cfg");
		~Config();

		GeneralSettings generalSettings;
		FluidSettings fluidSettings;
		TerrainSettings terrainSettings;
		SceneSettings sceneSettings;

		Ogre::ConfigFile* getCfg() { return mCfg; }

	private:

		Ogre::ConfigFile *mCfg;

		void Config::loadConfig();
		void Config::loadSceneConfig();
		void Config::loadFluidConfig();
		void Config::loadTerrainConfig();

	};

}

#endif

