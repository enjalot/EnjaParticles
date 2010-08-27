#include "SnowTerrain.h"

#define TERRAIN_PAGE_MIN_X 0
#define TERRAIN_PAGE_MIN_Y 0
#define TERRAIN_PAGE_MAX_X 0
#define TERRAIN_PAGE_MAX_Y 0

#define TERRAIN_FILE_PREFIX String("testTerrain")
#define TERRAIN_FILE_SUFFIX String("dat")

using namespace Ogre;
using namespace OgreBites;

namespace SnowSim
{
	SnowTerrain::SnowTerrain(SnowSim::Config *snowConfig)
	: mSnowConfig(snowConfig)
	, mTerrainGlobals(NULL)
	, mTerrainGroup(NULL)
	, mUpdateCountDown(0)
	, mTerrainPos(0,0,0)
	, mTerrainsImported(false)
	, mTerrainSize(mSnowConfig->terrainSettings.size)
	, mTerrainWorldSize(mSnowConfig->terrainSettings.worldSize)
	, mTerrainWorldScale(mSnowConfig->terrainSettings.worldScale)
	, mUpdateRate(1.0f / 20.0)
	, mSceneMgr(NULL)
	, mDebugNormalsNode(NULL)
	, mDebugNormalsManualObject(NULL)
	//, mSceneCreated(false)
	{
		mTerrainPos = snowConfig->sceneSettings.terrainPosition;

	// 	Ogre::Image combined;
	// 
	// 	combined.loadTwoImagesAsRGBA("terrain_1024_alpine3_shader_base.bmp", "terrain_1024_alpine3_shader_base_SPEC.bmp", 
	// 		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::PF_BYTE_RGBA);
	// 	combined.save("terrain_1024_alpine3_shader_base_diffusespecular.png");
	// 
	// 	combined.loadTwoImagesAsRGBA("terrain_1024_alpine3_shader_base_NORM.tga", "terrain_1024_alpine3_shader_base_DISP.bmp", 
	// 		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::PF_BYTE_RGBA);
	// 	combined.save("terrain_1024_alpine3_shader_base_normalheight.png");
	// 
	// 
	// 	combined.loadTwoImagesAsRGBA("terrain_1024_alpine3_shader_white.bmp", "terrain_1024_alpine3_shader_white_SPEC.bmp", 
	// 		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::PF_BYTE_RGBA);
	// 	combined.save("terrain_1024_alpine3_shader_white_diffusespecular.png");
	// 
	// 	combined.loadTwoImagesAsRGBA("terrain_1024_alpine3_shader_white_NORM.tga", "terrain_1024_alpine3_shader_white_DISP.bmp", 
	// 		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::PF_BYTE_RGBA);
	// 	combined.save("terrain_1024_alpine3_shader_white_normalheight.png");

		// Update terrain at max 20fps
		

		mTerrainGlobals = OGRE_NEW TerrainGlobalOptions();

	}

	SnowTerrain::~SnowTerrain()
	{
		OGRE_DELETE mTerrainGlobals;
	}

	void SnowTerrain::destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr)
	{
		OGRE_DELETE mTerrainGroup;
	}

	Ogre::TerrainMaterialGeneratorA::SM2Profile* SnowTerrain::getMaterialProfile()
	{
		return static_cast<TerrainMaterialGeneratorA::SM2Profile*>(mTerrainGlobals->getDefaultMaterialGenerator()->getActiveProfile());
	}

	//-------------------------------------------------------------------------------------
	void SnowTerrain::createScene(Ogre::SceneManager* sceneMgr, Light* terrainLight)
	{
		mSceneMgr = sceneMgr;

		mTerrainGroup = OGRE_NEW TerrainGroup(mSceneMgr, Terrain::ALIGN_X_Z, mTerrainSize, mTerrainWorldSize);
		mTerrainGroup->setFilenameConvention(TERRAIN_FILE_PREFIX, TERRAIN_FILE_SUFFIX);
		//mTerrainGroup->setOrigin(mTerrainPos);
		mTerrainGroup->setOrigin(mTerrainPos + Ogre::Vector3(mTerrainWorldSize / 2, 0, mTerrainWorldSize / 2));

		// Configure global
		mTerrainGlobals->setMaxPixelError(20); // set to 1 if using 1 unit/1 metre (ie pagesize of 1024)
		//mTerrainGlobals->setCompositeMapDistance(3000);
		mTerrainGlobals->setCompositeMapDistance(100000);
		//mTerrainGlobals->setUseRayBoxDistanceCalculation(true);
		//mTerrainGlobals->getDefaultMaterialGenerator()->setDebugLevel(1);
		mTerrainGlobals->setLightMapSize(256);

		//matProfile->setLightmapEnabled(false);

		// Important to set these so that the terrain knows what to use for derived (non-realtime) data
		mTerrainGlobals->setLightMapDirection(terrainLight->getDerivedDirection());
		mTerrainGlobals->setCompositeMapAmbient(mSceneMgr->getAmbientLight());
		//mTerrainGlobals->setCompositeMapAmbient(ColourValue::Red);
		mTerrainGlobals->setCompositeMapDiffuse(terrainLight->getDiffuseColour());

		// Configure default import settings for if we use imported image
		Terrain::ImportData& defaultimp = mTerrainGroup->getDefaultImportSettings();
		defaultimp.terrainSize = mSnowConfig->terrainSettings.size;
		defaultimp.worldSize = mTerrainWorldSize;
		defaultimp.inputScale = mSnowConfig->terrainSettings.worldScale;
		defaultimp.inputBias = 0;

		defaultimp.minBatchSize = 33;
		defaultimp.maxBatchSize = 65;

		// textures
		TextureLayerFileList diffSpecList = mSnowConfig->terrainSettings.textureLayerDiffSpecFileList;
		TextureLayerFileList normalHeightList = mSnowConfig->terrainSettings.textureLayerNormalHeightFileList;

		size_t textureLayers = std::max(diffSpecList.size(), normalHeightList.size());

		defaultimp.layerList.resize(std::max((size_t)1,textureLayers));
		for(int i = 0; i < textureLayers; i++)
		{
			defaultimp.layerList[i].worldSize =  mTerrainWorldSize;
			if(diffSpecList.size() >= i)
				defaultimp.layerList[i].textureNames.push_back(diffSpecList[i]);

			if(normalHeightList.size() >= i)
				defaultimp.layerList[i].textureNames.push_back(normalHeightList[i]);
		}

		for (long x = TERRAIN_PAGE_MIN_X; x <= TERRAIN_PAGE_MAX_X; ++x)
			for (long y = TERRAIN_PAGE_MIN_Y; y <= TERRAIN_PAGE_MAX_Y; ++y)
				defineTerrain(x, y, mSnowConfig->terrainSettings.flat);

		// sync load since we want everything in place when we start
		mTerrainGroup->loadAllTerrains(false);

		if (mTerrainsImported)
		{
			TerrainGroup::TerrainIterator ti = mTerrainGroup->getTerrainIterator();
			while(ti.hasMoreElements())
			{
				Terrain* t = ti.getNext()->instance;
				initBlendMaps(t);
			}
		}

		mTerrainGroup->freeTemporaryResources();

		// create/show debug normals
		if(mSnowConfig->terrainSettings.showDebugNormals)
		{
			mDebugNormalsManualObject = createDebugNormals(mSceneMgr);
			mDebugNormalsNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
			mDebugNormalsNode->attachObject(mDebugNormalsManualObject);
		}

	}

	bool SnowTerrain::frameRenderingQueued(const Ogre::FrameEvent& evt)
	{
		if(!mTerrainGroup) return true;

		if (mUpdateCountDown > 0)
		{
			mUpdateCountDown -= evt.timeSinceLastFrame;
			if (mUpdateCountDown <= 0)
			{
				mTerrainGroup->update();
				mUpdateCountDown = 0;

			}

		}
	// 	if (mTerrainGroup->isDerivedDataUpdateInProgress())
	// 	{
	// 		mTrayMgr->moveWidgetToTray(mInfoLabel, TL_TOP, 0);
	// 		mInfoLabel->show();
	// 		if (mTerrainsImported)
	// 		{
	// 			mInfoLabel->setCaption("Building terrain, please wait...");
	// 		}
	// 		else
	// 		{
	// 			mInfoLabel->setCaption("Updating textures, patience...");
	// 		}
	// 
	// 	}
	// 	else
	// 	{
	// 		mTrayMgr->removeWidgetFromTray(mInfoLabel);
	// 		mInfoLabel->hide();
	// 		if (mTerrainsImported)
	// 		{
	// 			saveTerrains(true);
	// 			mTerrainsImported = false;
	// 		}
	// 	}
		return true;
	}

	void SnowTerrain::defineTerrain(long x, long y, bool flat)
	{
		if (flat)
		{
			mTerrainGroup->defineTerrain(x, y, 0.0f);
		}
		else
		{
			if(StringUtil::endsWith(mSnowConfig->terrainSettings.heightDataFile, "dat"))
			{
				mTerrainGroup->defineTerrain(x, y, mSnowConfig->terrainSettings.heightDataFile);
				mTerrainGroup->loadTerrain(x,y, true);
			}
			else 
			{
				Image *img = new Image();

				if(StringUtil::endsWith(mSnowConfig->terrainSettings.heightDataFile, "raw"))
				{
					Ogre::DataStreamPtr stream = Ogre::ResourceGroupManager::getSingleton().openResource(mSnowConfig->terrainSettings.heightDataFile, ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
					size_t size = stream->size();
					img->loadRawData(stream, mTerrainSize-1, mTerrainSize-1, 1, PixelFormat::PF_FLOAT32_R);

					// height data must be square			
					assert(img->getWidth() == img->getHeight());

					// resize the height data if it's the wrong size
					if(img->getWidth() != mTerrainSize)
						img->resize(mTerrainSize, mTerrainSize);

					//img->flipAroundY();

					//mTerrainGroup->defineTerrain(x, y, (float*)img->getPixelBox().data);
					mTerrainGroup->defineTerrain(x, y, img);
					mTerrainGroup->loadTerrain(x,y, true);

				}
				else if(StringUtil::endsWith(mSnowConfig->terrainSettings.heightDataFile, "png")||StringUtil::endsWith(mSnowConfig->terrainSettings.heightDataFile, "bmp"))
				{
 					img->load(mSnowConfig->terrainSettings.heightDataFile, ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);		

					// height data must be square			
					assert(img->getWidth() == img->getHeight());

					// resize the height data if it's the wrong size
					if(img->getWidth() != mTerrainSize)
						img->resize(mTerrainSize, mTerrainSize);

					mTerrainGroup->defineTerrain(x, y, img);
					mTerrainGroup->loadTerrain(x,y, true);
				}


			}
		
			
			mTerrainsImported = true;
		}
	}


	void SnowTerrain::initBlendMaps(Terrain* terrain)
	{
		TextureLayerFileList blendImages = mSnowConfig->terrainSettings.textureBlendFileList;

		// load those blendmaps into the layers
		for(int j = 0;j < terrain->getLayerCount();j++)
		{
			// skip first layer
			if(j==0)
				continue;

			// no blend map for this layer
			if(blendImages.size() >= j && blendImages[j].length() == 0)
				continue;

			Ogre::TerrainLayerBlendMap *blendmap = terrain->getLayerBlendMap(j);
			Ogre::Image img;

			img.load(blendImages[j],"General");
			int blendmapsize = terrain->getLayerBlendMapSize();
			if(img.getWidth() != blendmapsize)
				img.resize(blendmapsize, blendmapsize);

			float *ptr = blendmap->getBlendPointer();
			Ogre::uint8 *data = static_cast<Ogre::uint8*>(img.getPixelBox().data);

			for(int bp = 0;bp < blendmapsize * blendmapsize;bp++)
				ptr[bp] = static_cast<float>(data[bp]) / 255.0f;

			blendmap->dirty();
			blendmap->update();
		}
	}

	void SnowTerrain::SaveTerrains(bool onlyIfModified)
	{
		if(!mTerrainGroup) return;

		//mTerrainGroup->saveAllTerrains(onlyIfModified);
		mTerrainGroup->saveAllTerrains(true);
	}


	bool SnowTerrain::keyPressed (const OIS::KeyEvent &evt)
	{
		switch (evt.key)
		{
			case OIS::KC_N:
				mSnowConfig->terrainSettings.showDebugNormals = !mSnowConfig->terrainSettings.showDebugNormals;

				if(mSceneMgr != NULL)
				{
					if(mDebugNormalsNode == NULL)
					{
						mDebugNormalsManualObject = createDebugNormals(mSceneMgr);
						mDebugNormalsNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
					}

					if(!mSnowConfig->terrainSettings.showDebugNormals)
						mDebugNormalsNode->detachObject(mDebugNormalsManualObject);
					else
						mDebugNormalsNode->attachObject(mDebugNormalsManualObject);
					break;
				}
		}
		return true;
	}


	void SnowTerrain::dumpTextures()
	{
		if(!mTerrainGroup) return;

		TerrainGroup::TerrainIterator ti = mTerrainGroup->getTerrainIterator();
		while (ti.hasMoreElements())
		{
			Ogre::uint32 tkey = ti.peekNextKey();
			TerrainGroup::TerrainSlot* ts = ti.getNext();
			if (ts->instance && ts->instance->isLoaded())
			{
				ts->instance->_dumpTextures("terrain_" + StringConverter::toString(tkey), ".png");
			}
		}
	}


	float* SnowTerrain::getTerrainHeightData()
	{
		Terrain *t =  mTerrainGroup->getTerrain(0,0);

		// Get terrain height data using official method
		float* terrainHeightData = t->getHeightData();

		return terrainHeightData;
	}

	Vector4* SnowTerrain::getTerrainNormalData()
	{
		PixelBox* terrainNormals;

		// load from normals file
		if(mSnowConfig->terrainSettings.normalsDataFile.length() > 0)
		{
			// get terrain normal data using image
			Ogre::Image img;
			img.load(mSnowConfig->terrainSettings.normalsDataFile,  "General");
			//img.flipAroundY();
			//img.flipAroundX();

			size_t size = img.getWidth();
			assert(img.getWidth() == img.getHeight());

			if (img.getWidth() != mTerrainSize || img.getHeight() != mTerrainSize)
				img.resize(mTerrainSize, mTerrainSize);

			terrainNormals = &img.getPixelBox();

			Vector4* floats = convertNormalsToFloats(terrainNormals, true);
			//OGRE_FREE(terrainNormals->data, Ogre::MEMCATEGORY_GENERAL);
			
			// need to swap z and y vector due to different vertical axis in normal map and world space!
			for(size_t i = 0;i<mTerrainSize*mTerrainSize;i++)
			{
				Vector4 v = floats[i];
				floats[i].z = v.y;
				floats[i].y = v.z;

			}
			return floats;
		}
		else
		{
			// need to wait until terrain is loaded
			while (getTerrain()->isDerivedDataUpdateInProgress())
			{
				// we need to wait for this to finish
				OGRE_THREAD_SLEEP(50);
				Root::getSingleton().getWorkQueue()->processResponses();
			}

			// Get terrain normal data using official method
			//terrainNormals = getTerrain()->calculateNormals(Ogre::Rect(0,0,mTerrainSize,mTerrainSize),Rect(0,0,mTerrainSize,mTerrainSize));
			Ogre::Image img;
			getTerrain()->getTerrainNormalMap()->convertToImage(img);
			//img.flipAroundY();
			img.flipAroundX();
			//img.save("test_normals.bmp");
			terrainNormals = &img.getPixelBox();

			Vector4* floats = convertNormalsToFloats(terrainNormals, true);
			//OGRE_FREE(terrainNormals->data, Ogre::MEMCATEGORY_GENERAL);
			return floats;
		}


		
	}

	int SnowTerrain::getTerrainSize()
	{
		return mTerrainSize;
	}

	Real SnowTerrain::getTerrainWorldSize()
	{
		return mTerrainWorldSize;
	}

	Terrain* SnowTerrain::getTerrain()
	{
		if(!mTerrainGroup) return NULL;

		Terrain *t =  mTerrainGroup->getTerrain(0,0);
		return t;

		TerrainGroup::TerrainIterator ti = mTerrainGroup->getTerrainIterator();
		while (ti.hasMoreElements())
		{
			Ogre::uint32 tkey = ti.peekNextKey();
			TerrainGroup::TerrainSlot* ts = ti.getNext();
			if (ts->instance && ts->instance->isLoaded())
			{

				float* heights = ts->instance->getHeightData();

				//PixelBox* pBox = ts->instance->calculateNormals());
				TexturePtr texturePtr = ts->instance->getTerrainNormalMap();
				HardwarePixelBufferSharedPtr buf = texturePtr->getBuffer();

				size_t bytes = buf->getSizeInBytes();
				size_t h = buf->getHeight();
				size_t w = buf->getWidth();
				size_t d = buf->getDepth();
				PixelFormat f = PF_BYTE_RGB;//buf->getFormat();


				uint8* tmpData = (uint8*)OGRE_MALLOC(w * h * 3, MEMCATEGORY_GENERAL);
				memset(tmpData,0,w*h*3);
				PixelBox pBox(w, h, d, f, tmpData);
				buf->blitToMemory(pBox);
				OGRE_FREE(tmpData, MEMCATEGORY_GENERAL);
				
			}
		}
		return NULL;
	}
	Vector4* SnowTerrain::convertNormalsToFloats(PixelBox* terrainNormals, bool compressed)
	{
		const size_t srcPixelSize = PixelUtil::getNumElemBytes(terrainNormals->format);
		const size_t dstPixelSize = PixelUtil::getNumElemBytes(PF_FLOAT32_RGBA);
		size_t w,h,d;
		w = terrainNormals->getWidth();
		h = terrainNormals->getHeight();
		d = terrainNormals->getDepth();

		assert(terrainNormals->getWidth() == mTerrainSize);
		size_t terrainNormalSize = terrainNormals->getWidth()*terrainNormals->getHeight();
		Vector4* terrainNormalDataCorrected = OGRE_ALLOC_T(Vector4, terrainNormalSize, MEMCATEGORY_GENERAL);

		size_t i = 0; size_t j = 0;
		uint8* pixelsBuffer = static_cast<uint8*>(terrainNormals->data);
		for(; i < terrainNormalSize * srcPixelSize && j < terrainNormalSize * sizeof(Vector4); i+=srcPixelSize)
		{
			uint8 r,g,b,a;
			PixelUtil::unpackColour(&r,&g,&b,&a, terrainNormals->format, static_cast<void*>(&pixelsBuffer[i]));
			float fr,fg,fb;
			if(compressed)
			{			
				//(signed) float packed/compressed into uint8, unpack
				fr = ((float)(r))/(0.5f * 255.0f) - 1.0f;
				fg = ((float)(g))/(0.5f * 255.0f) - 1.0f;
				fb = ((float)(b))/(0.5f * 255.0f) - 1.0f;
			}
			else
			{
				fr = ((float)(r))/255.0f;
				fg = ((float)(g))/255.0f;
				fb = ((float)(b))/255.0f;
			}
			Vector3 v = Vector3(fr, fg, fb);
			v.normalise();
			terrainNormalDataCorrected[j++] = Vector4(v);
		}

		return terrainNormalDataCorrected;
	}

	ManualObject* SnowTerrain::createDebugNormals(Ogre::SceneManager* mSceneMgr)
	{
		ManualObject* manual = mSceneMgr->createManualObject("NormalsDebug");

		float *heights = getTerrainHeightData();
		Vector4 *normals = getTerrainNormalData();
		manual->begin("BaseWhiteNoLighting", RenderOperation::OT_LINE_LIST);

		int terrainSize = getTerrain()->getSize();
		int terrainWorldSize = getTerrain()->getWorldSize();

		for(int z = 0; z < terrainSize; ++z)
		{
			for(int x = 0; x < terrainSize; ++x)
			{
				int i = ((terrainSize)*(terrainSize) ) - (z+1)*terrainSize + (x+1);

				Vector3 n = Vector3(normals[i].x, normals[i].y, normals[i].z);

				float h = heights[i];

				Real factor = (Real)terrainSize - 1.0f;
				Real invFactor = 1.0f / factor;

				float mScale =  terrainWorldSize / (Real)(terrainSize);

				Real mBase = -terrainWorldSize * 0.5;
				Vector3 mPos = Ogre::Vector3(terrainWorldSize * 0.5, 0, terrainWorldSize * 0.5);

				Vector3 worldPos;

				worldPos.x = x * mScale + mBase + mPos.x;
				worldPos.y = h + mPos.y;
				worldPos.z = z * mScale + mBase + mPos.z;

				// convert back to "normal map" colors (0/1 float instead of -1/1 float, also rgb=>xzy)
				manual->colour((n.x + 1)*0.5, (n.z + 1)*0.5, (n.y + 1)*0.5);

				// draw line
				manual->position(worldPos);
				manual->position(worldPos + 3*n);

				//manual->index(i);
				//manual->index(1);

			}
		}
		manual->end();

		//OGRE_FREE(normals);
		return manual;
	}
}