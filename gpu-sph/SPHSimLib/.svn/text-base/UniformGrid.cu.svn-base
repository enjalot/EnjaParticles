#include "UniformGrid.cuh"
#include "CudaUtils.cuh"

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

__device__ __constant__	GridParams		cGridParams;

#include "K_UniformGrid_Hash.cu"

using namespace SimLib;

UniformGrid::UniformGrid(SimLib::SimCudaAllocator* simCudaAllocator)
	: mSimCudaAllocator(simCudaAllocator)
	, mAlloced(false)
	, mUseCUDPPSort(true)
{
	mGPUTimer = new ocu::GPUTimer();

	mGridParticleBuffers = new BufferManager<UniformGridBuffers>(mSimCudaAllocator);
	mGridCellBuffers = new BufferManager<UniformGridBuffers>(mSimCudaAllocator);

	mGridParticleBuffers->SetBuffer(SortHashes,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridParticleBuffers->SetBuffer(SortIndexes,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridCellBuffers->SetBuffer(CellIndexesStart,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridCellBuffers->SetBuffer(CellIndexesStop,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));

}

UniformGrid::~UniformGrid()
{
	Free();

	delete mGPUTimer;
	delete mGridCellBuffers;
	delete mGridParticleBuffers;
}

void UniformGrid::Alloc(uint numParticles, float cellWorldSize, float gridWorldSize)
{
	if(mAlloced)
		Free();

	CalculateGridParameters(cellWorldSize, gridWorldSize);

	mNumParticles = numParticles;

	// only need X bits precision for the radix sort.. (256^3 volume ==> 24 bits precision)
	mSortBitsPrecision = (uint)ceil(log2(dGridParams.grid_res.x*dGridParams.grid_res.y*dGridParams.grid_res.z));
	//	assert(mSortBitsPrecision => 4 && mSortBitsPrecision <= 32);

	// number of cells is given by the resolution (~how coarse the grid of the world is)
	mNumCells = (int)ceil(dGridParams.grid_res.x*dGridParams.grid_res.y*dGridParams.grid_res.z);


	// Allocate grid buffers
	mGridParticleBuffers->AllocBuffers(mNumParticles);
	mGridCellBuffers->AllocBuffers(mNumCells);

	// Allocate the radix sorter
	if(mUseCUDPPSort)
	{
		// Create the CUDPP radix sort
		CUDPPConfiguration sortConfig;
		sortConfig.algorithm = CUDPP_SORT_RADIX;
		sortConfig.datatype = CUDPP_UINT;
		sortConfig.op = CUDPP_ADD;
		sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
		cudppPlan(&m_sortHandle, sortConfig, mNumParticles, 1, 0);
	}
	else
	{
		mRadixSorter = new RadixSort(mNumParticles * sizeof(uint));
	}

	//Copy the grid parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cGridParams, &dGridParams, sizeof(GridParams) ) );
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	mAlloced = true;
}

void UniformGrid::Free()	
{
	if(!mAlloced)
		return;

	if(mUseCUDPPSort)
	{
		cudppDestroyPlan(m_sortHandle);	m_sortHandle=NULL;
	}
	else
	{
		delete mRadixSorter; mRadixSorter = NULL;
	}

	mGridParticleBuffers->FreeBuffers();
	mGridCellBuffers->FreeBuffers();

	mAlloced = false;
}

void UniformGrid::Clear()
{
// 	mGridCellBuffers->ClearBuffers();
// 	mGridParticleBuffers->ClearBuffers();
}

void UniformGrid::CalculateGridParameters(float cellWorldSize, float gridWorldSize)
{
	// GRID SETUP

	// Ideal grid "cell" size (gs) = 2 * smoothing length	.. then we can use 8 cell checker
	// however ... atm we use particles 27 cell checker, so cell size must be equal to smoothing length
	dGridParams.grid_min = make_float3(0, 0, 0);
	dGridParams.grid_max = dGridParams.grid_min + (float)gridWorldSize;

	dGridParams.grid_size = make_float3(
		dGridParams.grid_max.x-dGridParams.grid_min.x, 
		dGridParams.grid_max.y-dGridParams.grid_min.y, 
		dGridParams.grid_max.z-dGridParams.grid_min.z);

	dGridParams.grid_res = make_float3(
		ceil(dGridParams.grid_size.x / cellWorldSize), 
		ceil(dGridParams.grid_size.y / cellWorldSize),
		ceil(dGridParams.grid_size.z / cellWorldSize));

	// Adjust grid size to multiple of cell size	
	dGridParams.grid_size.x = dGridParams.grid_res.x * cellWorldSize;				
	dGridParams.grid_size.y = dGridParams.grid_res.y * cellWorldSize;
	dGridParams.grid_size.z = dGridParams.grid_res.z * cellWorldSize;

	dGridParams.grid_delta.x = dGridParams.grid_res.x / dGridParams.grid_size.x;
	dGridParams.grid_delta.y = dGridParams.grid_res.y / dGridParams.grid_size.y;
	dGridParams.grid_delta.z = dGridParams.grid_res.z / dGridParams.grid_size.z;
};


float UniformGrid::Hash(bool doTiming, float_vec* dParticlePositions, uint numParticles)
{
//	assert(mNumParticles == numParticles);

	// clear old hash values
	mGridParticleBuffers->Get(SortHashes)->Memset(0);

	int threadsPerBlock;

	// Used 14 registers, 64+16 bytes smem, 144 bytes cmem[0]
	threadsPerBlock = 128;

#ifdef SPHSIMLIB_FERMI
	threadsPerBlock = 192;
#endif

	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);

	while(numBlocks >= 64*1024)
	{
		cout << "ALERT: have to rescale threadsPerBlock due to too large grid size >=65536\n";
		threadsPerBlock += 32;
		computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	}

	GridData dGridData = GetGridData();

	if(doTiming)
	{
		mGPUTimer->start();
	}

	// hash each particle according to spatial position (cell in grid volume)
	K_Grid_Hash<<< numBlocks, numThreads>>> (
		mNumParticles,
		dParticlePositions, 
		dGridData
		);	

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float UniformGrid::Sort(bool doTiming)
{
	if(doTiming)
	{
		mGPUTimer->start();
	}

	if(mUseCUDPPSort)
	{
		cudppSort(
			m_sortHandle, 
			mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(), 
			mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>(),
			mSortBitsPrecision, 
			mNumParticles);
	}
	else
	{
		mRadixSorter->sort(
			mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(),
			mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>(),
			mNumParticles,
			mSortBitsPrecision);
	}



	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}
