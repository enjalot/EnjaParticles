#define STRINGIFY(A) #A

std::string hash_program_source = STRINGIFY(

// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

//#include "K_UniformGrid_Utils.cu"

//----------------------------------------------------------------------
// find the grid cell from a position in world space
// WHY static?
//static int3 calcGridCell(float3 const &p, float3 grid_min, float3 grid_delta)
int3 calcGridCell(float3 const &p, float3 grid_min, float3 grid_delta)
{
	// subtract grid_min (cell position) and multiply by delta
	return make_int3((p-grid_min) * grid_delta);
}

//----------------------------------------------------------------------
// Calculate a grid hash value for each particle

__kernel void K_Grid_Hash (
							   unsigned int			numParticles,
							   __global float_vec*	dParticlePositions,	
							   GridData				dGridData
							   )
{			
	// particle index
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	// particle position
	float4 p = dParticlePositions[index];

	// get address in grid
	int3 gridPos = UniformGridUtils::calcGridCell(make_float3(p), cGridParams.grid_min, cGridParams.grid_delta);
	uint hash = UniformGridUtils::calcGridHash<true>(gridPos, cGridParams.grid_res);

	// store grid hash and particle index
	dGridData.sort_hashes[index] = hash;
	dGridData.sort_indexes[index] = index;

}
//----------------------------------------------------------------------


);
