#ifndef _UNIFORM_GRID_UTILS_CL_
#define _UNIFORM_GRID_UTILS_CL_

// TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 

//----------------------------------------------------------------------

// Template parameters
#define D Step1::Data
#define O SPHNeighborCalc<Step1::Calc, Step1::Data>

// copied from SPHSimLib code
#ifdef USE_TEX
//#define FETCH(a, t, i) tex1Dfetch(t##_tex, i)
#define FETCH(t, i) tex1Dfetch(t##_tex, i)
#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_MATRIX3(a,t,i) tex1DfetchMatrix3(t##_tex,i)
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
#else
//#define FETCH(a, t, i) a.t[i]
#define FETCH(t, i) t[i]
//#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_NOTEX(t, i) t[i]
//#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_FLOAT3(t,i) make_float3(FETCH(t,i))
#define FETCH_MATRIX3(a,t,i) a.t[i]
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
//#define FETCH(a, t, i) (a + __mul24(i,sizeof(a)) + (void*)offsetof(a, t))
#endif


struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
};


	//--------------------------------------------------------------
	// Iterate over particles found in the nearby cells (including cell of position_i)
	//template<class O, class D>
	//static __device__ void IterateParticlesInCell(
	void IterateParticlesInCell(
		D 					&data, 
		//int3 const		&cellPos,
		__constant int4 	&cellPos,
		__constant uint 	&index_i, 
		//float3 const		&position_i, 
		__constant float4 	&position_i, 
		//GridData const	&dGridData,
		__global__ int* cell_indexes_start,
		__global__ int* cell_indexes_end
    )
	{
		// get hash (of position) of current cell
		//volatile uint cellHash = UniformGridUtils::calcGridHash<true>(cellPos, cGridParams.grid_res);
		// wrap edges (true)
		uint cellHash = calcGridHash(cellPos, cGridParams.grid_res, true);

		// get start/end positions for this cell/bucket
		//uint startIndex	= FETCH_NOTEX(dGridData,cell_indexes_start,cellHash);
		//volatile uint startIndex = FETCH(dGridData,cell_indexes_start,cellHash);
		uint startIndex = FETCH(cell_indexes_start,cellHash);

		// check cell is not empty
		if (startIndex != 0xffffffff) 
		{	   
			//uint endIndex = FETCH_NOTEX(dGridData,cell_indexes_end,cellHash);
			//volatile uint endIndex = FETCH(dGridData, cell_indexes_end, cellHash);
			uint endIndex = FETCH(cell_indexes_end, cellHash);

			// iterate over particles in this cell
			for(uint index_j=startIndex; index_j < endIndex; index_j++) 
			{			
				O::ForPossibleNeighbor(data, index_i, index_j, position_i);
			}
		}
	}

	//--------------------------------------------------------------
	// Iterate over particles found in the nearby cells (including cell of position_i)
	//template<class O, class D>
	//static __device__ void IterateParticlesInNearbyCells(
	static void IterateParticlesInNearbyCells(
		D 					&data, 
		//uint const		&index_i, 
		uint __constant		&index_i, 
		//float3 const		&position_i, 
		__constant float4   &position_i, 
		//GridData const	&dGridData
		__constant struct GridParams& cGridParams)
		__global int* 		cell_indexes_start,
		__global int* 		cell_indexes_end)
	{
		// How to chose which PreCalc to use? 
		O::PreCalc(data, index_i);

		// get cell in grid for the given position
		//volatile int3 cell = UniformGridUtils::calcGridCell(position_i, cGridParams.grid_min, cGridParams.grid_delta);
		int4 cell = calcGridCell(position_i, cGridParams.grid_min, cGridParams.grid_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost 
		for(int z=cell.z-1; z<=cell.z+1; ++z) {
			for(int y=cell.y-1; y<=cell.y+1; ++y) {
				for(int x=cell.x-1; x<=cell.x+1; ++x) {
					IterateParticlesInCell<O,D>(data, make_int3(x,y,z), index_i, position_i, dGridData);
				}
			}
		}

		O::PostCalc(data, index_i);
	}

	//----------------------------------------------------------------------
	// Iterate over particles found in the neighbor list
	//template<class O, class D>
	static __device__ void IterateParticlesInNearbyCells(
		D 					&data, 
		__constant uint 	&index_i, 
		__constant float3 	&position_i, 
		__constant NeighborList &dNeighborList
		)
	{
		O::PreCalc(data, index_i);

		// iterate over particles in neighbor list
		for(uint counter=0; counter < dNeighborList.MAX_NEIGHBORS; counter++) 
		{
			//const uint index_j = FETCH(dNeighborList,neighbors, index_i*dNeighborList.neighbors_pitch+counter);
			__constant uint index_j = FETCH_NOTEX(neighbors, index_i*dNeighborList.MAX_NEIGHBORS+counter);			

			// no more neighbors for this particle
			if(index_j == 0xffffffff)
				break;

			O::ForPossibleNeighbor(data, index_i, index_j, position_i);

		}

		O::PostCalc(data, index_i);
	}

//--------------------------------------------------------------


#endif
