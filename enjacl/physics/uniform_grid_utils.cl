#ifndef _UNIFORM_GRID_UTILS_CL_
#define _UNIFORM_GRID_UTILS_CL_

// TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 

//----------------------------------------------------------------------

// Template parameters
//#define D Step1::Data
#define D float
#define O SPHNeighborCalc<Step1::Calc, Step1::Data>

#undef USE_TEX

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
#define FETCH_VAR(t, i, ivar) t[i+ivar*numParticles]
//#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_NOTEX(t, i) t[i]
//#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_FLOAT3(t,i) make_float3(FETCH(t,i))
#define FETCH_MATRIX3(a,t,i) a.t[i]
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
//#define FETCH(a, t, i) (a + __mul24(i,sizeof(a)) + (void*)offsetof(a, t))
#endif

#if 0
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
int4 calcGridCell(float4 p, float4 grid_min, float4 grid_delta)
{
	// subtract grid_min (cell position) and multiply by delta
	//return make_int4((p-grid_min) * grid_delta);

	//float4 pp = (p-grid_min)*grid_delta;
	float4 pp;
	pp.x = (p.x-grid_min.x)*grid_delta.x;
	pp.y = (p.y-grid_min.y)*grid_delta.y;
	pp.z = (p.z-grid_min.z)*grid_delta.z;
	pp.w = (p.w-grid_min.w)*grid_delta.w;

	int4 ii;
	ii.x = (int) pp.x;
	ii.y = (int) pp.y;
	ii.z = (int) pp.z;
	ii.w = (int) pp.w;
	return ii;
}

	//--------------------------------------------------------------
uint calcGridHash(int4 gridPos, float4 grid_res, __constant bool wrapEdges)
{
	// each variable on single line or else STRINGIFY DOES NOT WORK
	int gx;
	int gy;
	int gz;

	if(wrapEdges) {
		int gsx = (int)floor(grid_res.x);
		int gsy = (int)floor(grid_res.y);
		int gsz = (int)floor(grid_res.z);

//          //power of 2 wrapping..
//          gx = gridPos.x & gsx-1;
//          gy = gridPos.y & gsy-1;
//          gz = gridPos.z & gsz-1;

		// wrap grid... but since we can not assume size is power of 2 we can't use binary AND/& :/
		gx = gridPos.x % gsx;
		gy = gridPos.y % gsy;
		gz = gridPos.z % gsz;
		if(gx < 0) gx+=gsx;
		if(gy < 0) gy+=gsy;
		if(gz < 0) gz+=gsz;
	} else {
		gx = gridPos.x;
		gy = gridPos.y;
		gz = gridPos.z;
	}


	//We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
	//this space filling curve is then simply:
	// index = x + y*width + z*width*height
	//This means that we process the grid structure in "depth slice" order, and
	//each such slice is processed in row-column order.
	//return __mul24(__umul24(gz, grid_res.y), grid_res.x) + __mul24(gy, grid_res.x) + gx;

	return (gz*grid_res.y + gy) * grid_res.x + gx; 
}
#endif
	//--------------------------------------------------------------



#if 0
	// Iterate over particles found in the nearby cells (including cell of position_i)
	//template<class O, class D>
	//static __device__ void IterateParticlesInCell(
	void IterateParticlesInCell(
		__constant int4 	cellPos,
		__constant uint 	index_i, 
		__constant float4 	position_i, 
		__global int* 		cell_indexes_start,
		__global int* 		cell_indexes_end, 
		__constant struct GridParams* cGridParams
    )
	{
		// get hash (of position) of current cell
		//volatile uint cellHash = UniformGridUtils::calcGridHash<true>(cellPos, cGridParams.grid_res);
		// wrap edges (false)
		uint cellHash = calcGridHash(cellPos, cGridParams->grid_res, false);

		// get start/end positions for this cell/bucket
		//uint startIndex	= FETCH_NOTEX(dGridData,cell_indexes_start,cellHash);
		//volatile uint startIndex = FETCH(dGridData,cell_indexes_start,cellHash);
		uint startIndex = FETCH(cell_indexes_start,cellHash);

		// check cell is not empty
		if (startIndex != 0xffffffff) 
		{	   
			uint endIndex = FETCH(cell_indexes_end, cellHash);

			// iterate over particles in this cell
			for(uint index_j=startIndex; index_j < endIndex; index_j++) 
			{			
				// For now, nothing to loop over. ADD WHEN CODE WORKS. 
				//ForPossibleNeighbor(data, index_i, index_j, position_i);
				;
			}
		}
	}

	//--------------------------------------------------------------
	// Iterate over particles found in the nearby cells (including cell of position_i)
	//template<class O, class D>
	//static __device__ void IterateParticlesInNearbyCells(
	void IterateParticlesInNearbyCells(
		__global float4* vars_sorted,
		//__global float4*     force_sorted,
		//__global float4*     pressure_sorted,
		//__global float4*     density_sorted,
		//__global float4*     position_sorted,
		//D 					data, 
		__constant uint 	index_i, 
		//float3 const		&position_i, 
		__constant float4   position_i, 
		//GridData const	&dGridData
		__global int* 		cell_indices_start,
		__global int* 		cell_indices_end,
		__constant struct GridParams* cGridParams)
	{
		// How to chose which PreCalc to use? 
		// TODO LATER
		//PreCalc(data, index_i); // TODO

		// get cell in grid for the given position
		int4 cell = calcGridCell(position_i, cGridParams->grid_min, cGridParams->grid_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost 
		for(int z=cell.z-1; z<=cell.z+1; ++z) {
			for(int y=cell.y-1; y<=cell.y+1; ++y) {
				for(int x=cell.x-1; x<=cell.x+1; ++x) {
					int4 ipos;
					ipos.x = x;
					ipos.y = y;
					ipos.z = z;
					ipos.w = 1;
					IterateParticlesInCell(ipos, index_i, position_i, cell_indices_start, cell_indices_end, cGridParams);
				}
			}
		}

		// TO REMOVE
		//O::PostCalc(data, index_i);

		// TO DO LATER
		//PostCalc(data, index_i);
	}

#endif
	//----------------------------------------------------------------------
//--------------------------------------------------------------
__kernel void K_SumStep1(
				uint    numParticles,
				uint	nb_vars,
				__global float4* vars//,   // *** ERROR
				#if 0
				__global float4* sorted_vars,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
				__constant struct GridParams* cGridParams
				#endif
				)
{
    // particle index
    //uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint index = get_global_id(0);
    if (index >= numParticles) return;

	#if 0
    //Step1::Data data;
    //data.dParticleDataSorted = dParticleDataSorted;

	vars = sorted_vars;
	//force = force_sorted;
	//pressure = pressure_sorted;
	//density  = density_sorted;
	//position = position_sorted;

	// assume position is 0th variable
    float4 position_i = FETCH_VAR(vars, index, 0);
	#endif

    // Do calculations on particles in neighboring cells

    //IterateParticlesInNearbyCells(force_sorted, pressure_sorted, density_sorted, position_sorted, index, position_i, cell_indexes_start, cell_indexes_end, cGridParams);

	#if 0
    IterateParticlesInNearbyCells(sorted_vars, index, position_i, cell_indexes_start, cell_indexes_end, cGridParams);
	#endif

}

//--------------------------------------------------------------
#endif

