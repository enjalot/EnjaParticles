#ifndef __SimulationSystem_h__
#define __SimulationSystem_h__

#include "Config.h"

#include "SimCudaAllocator.h"
#include "SimBufferManager.h"
#include "SimCudaHelper.h"

#include "SimBase.cuh"

typedef unsigned int uint;

namespace SimLib
{
	enum ParticleSimType
	{
		SimulationSimpleSPH
		, SimulationDEM
		, SimulationSnowSPH
	};

	class SimulationSystem 
	{
	public:
		SimulationSystem(bool simpleSPH);
		~SimulationSystem();

		void SetExternalBuffer(SimLib::Sim::BaseBufferId id, SimBuffer* buffer);
		void RemoveExernalBuffer(SimLib::Sim::BaseBufferId id);

		void Init();
		void SetNumParticles(uint numParticles);
		void SetFluidPosition(float3 fluidWorldPosition) ;
		void Simulate(bool progress, bool gridWallCollisions);

		void PrintMemoryUse();

		void Clear();

		void SetScene(int scene);
		void SetTerrainData(float3 terrainPosition, float *terrainHeightData, float4* terrainNormalData, int terrainSize, float terrainWorldSize);

		void SetPrintTiming(bool enable) 
		{
			mCudaTiming = enable;
		}
		SimSettings* GetSettings() {return mParticleSim->GetSettings();}
		float GetParticleSize();

	private:
		SimCudaAllocator* mSimCudaAllocator;

		ParticleSimType mParticleSimType;

		bool mInitialized;
		bool mBuffersMapped;

		float3 mFluidWorldPosition;


		float3 hTerrainPosition;
		float* hTerrainData;
		int hTerrainSize;
		float hTerrainWorldSize;

		void Free();

		std::map<SimLib::Sim::BaseBufferId, SimBuffer*> mExternalSimBuffers;

		void SimulationSystem::FillTestData(int scene, float_vec* position, int numParticles, GridParams hGridParams);

		void mapRenderingBuffers();
		void unmapRenderingBuffers();
		int mSimulationSteps;
		// MISc

		// Buffer for particles
		uint terrainGeometryDim;

		bool mCudaTiming;
		bool mHaveTerrainData;

		SimLib::Sim::SimBase* mParticleSim;

		SimLib::Sim::TerrainData  dTerrainData;

	};

}

#endif