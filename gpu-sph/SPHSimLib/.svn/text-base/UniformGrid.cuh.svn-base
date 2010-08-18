#ifndef __UniformGrid_cuh__
#define __UniformGrid_cuh__

#include "SimCudaAllocator.h"
#include "radixsort.h"
#include "cudpp/cudpp.h"
#include <cutil.h>

#include "UniformGrid.cuh"
#include "K_Common.cuh"
#include "ParticleData.h"
#include "SimBufferManager.h"
#include "SimBufferCuda.h"
#include "timer.h"

typedef unsigned int uint;

enum UniformGridBuffers
{
	SortHashes,
	SortIndexes,
	CellIndexesStart,
	CellIndexesStop,
};

struct NeighborList
{
	int numParticles;
	int MAX_NEIGHBORS;
	uint* neighbors;
	// pitch, IN ELEMENTS, NOT BYTES
	size_t		neighbors_pitch;
};

struct GridParams
{
	float3			grid_size;
	float3			grid_min;
	float3			grid_max;

	// number of cells in each dimension/side of grid
	float3			grid_res;

	float3			grid_delta;
};

struct GridData
{
	uint* sort_hashes;			// particle hashes
	uint* sort_indexes;			// particle indices
	uint* cell_indexes_start;	// mapping between bucket hash and start index in sorted list
	uint* cell_indexes_end;		// mapping between bucket hash and end index in sorted list
};


class UniformGrid
{
public:
	UniformGrid(SimLib::SimCudaAllocator* SimCudaAllocator);
	~UniformGrid();

	void Alloc(uint numParticles, float cellWorldSize, float gridWorldSize);
	void Clear();
	void Free()	;

	float Hash(bool doTiming, float_vec* dParticlePositions, uint numParticles);
	float Sort(bool doTiming);

	GridData GetGridData(){
		GridData gridData;
		gridData.cell_indexes_start = mGridCellBuffers->Get(CellIndexesStart)->GetPtr<uint>();
		gridData.sort_hashes = mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>();
		gridData.sort_indexes = mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>();
		gridData.cell_indexes_end = mGridCellBuffers->Get(CellIndexesStop)->GetPtr<uint>();
		return gridData;
	}

	uint GetNumCells(){return mNumCells;}
	GridParams& GetGridParams(){return dGridParams;}

private:
	void CalculateGridParameters(float cellWorldSize, float gridWorldSize);

	SimLib::BufferManager<UniformGridBuffers> *mGridParticleBuffers;
	SimLib::BufferManager<UniformGridBuffers> *mGridCellBuffers;

	uint mNumParticles;
	uint mNumCells;

	bool mAlloced;

	ocu::GPUTimer *mGPUTimer;

	SimLib::SimCudaAllocator* mSimCudaAllocator;

	GridParams dGridParams;

	bool mUseCUDPPSort;
	RadixSort* mRadixSorter;
	CUDPPHandle m_sortHandle;

	int mSortBitsPrecision;

};

#endif