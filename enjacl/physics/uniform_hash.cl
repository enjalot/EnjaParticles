#define STRINGIFY(A) #A

std::string hash_program_source = STRINGIFY(

// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

//#include "K_UniformGrid_Utils.cu"

//----------------------------------------------------------------------
// find the grid cell from a position in world space
// WHY static?
//static int4 calcGridCell(float4 const &p, float4 grid_min, float4 grid_delta)
int4 calcGridCell(float4 p, float4 grid_min, float4 grid_delta)
{
	// subtract grid_min (cell position) and multiply by delta
	//return make_int4((p-grid_min) * grid_delta);
	float4 pp = (p-grid_min)*grid_delta;

	int4 ii;
	ii.x = pp.x;
	ii.y = pp.y;
	ii.z = pp.z;
	ii.w = pp.w;
	return ii;
}

//----------------------------------------------------------------------
//template <bool wrapEdges>
//static uint calcGridHash(int3 const &gridPos, float3 grid_res, __constant bool wrapEdges)
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

	//return  __mul24(__mul24(gz, (int) cGridParams.grid_res.y)+gy, (int) cGridParams.grid_res.x) + gx;

	//We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
	//this space filling curve is then simply:
	// index = x + y*width + z*width*height
	//This means that we process the grid structure in "depth slice" order, and
	//each such slice is processed in row-column order.
	//return __mul24(__umul24(gz, grid_res.y), grid_res.x) + __mul24(gy, grid_res.x) + gx;

	return (gz*grid_res.y + gy) * grid_res.x + gx; 
}

//----------------------------------------------------------------------
// Calculate a grid hash value for each particle


//  Have to make sure that the data associated with the pointers is on the GPU
//struct GridData
//{
//    uint* sort_hashes;          // particle hashes
//    uint* sort_indexes;         // particle indices
//    uint* cell_indexes_start;   // mapping between bucket hash and start index in sorted list
//    uint* cell_indexes_end;     // mapping between bucket hash and end index in sorted list
//};

struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
};


// comes from K_Grid_Hash
// CANNOT USE references to structures/classes as aruguments!
__kernel void hash(
		   unsigned int				numParticles,
		   __global float4*	  		dParticlePositions,	
		   __global uint* sort_hashes,
		   __global uint* sort_indexes,
		   __constant struct GridParams* cGridParams)
{
	// particle index
	//uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint index = get_global_id(0);
	if (index >= numParticles) return;

	// particle position
	float4 p = dParticlePositions[index];

	// get address in grid
	int4 gridPos = calcGridCell(p, cGridParams->grid_min, cGridParams->grid_delta);
	bool wrap_edges = false;
	uint hash = calcGridHash(gridPos, cGridParams->grid_res, wrap_edges);
	//hash = cGridParams->grid_res.x;

	// store grid hash and particle index

	sort_hashes[index] = hash;
	sort_indexes[index] = index;
}
//----------------------------------------------------------------------


);
