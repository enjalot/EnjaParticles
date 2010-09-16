#ifndef __NEIGHBORS_CL_K_
#define __NEIGHBORS_CL_K_

#include "cl_macros.h"


//----------------------------------------------------------------------
void ForNeighbor(__global float4*  var_sorted, 
				__constant uint index_i, 
				uint index_j, 
				float4 r, 
				float rlen, 
				float rlen_sq) {
// the density sum using Wpoly6 kernel
//data.sum_density += SPH_Kernels::Wpoly6::Kernel_Variable(cPrecalcParams.smoothing_length_pow2, r, rlen_sq);	
	;
}
//--------------------------------------------------


void ForPossibleNeighbor(__global float4* var_sorted, 
						__constant uint numParticles, 
						__constant uint index_i, 
						uint index_j, 
						__constant float4 position_i)
{
	// check not colliding with self
	if (index_j != index_i) {		
		// get the particle info (in the current grid) to test against
		//float3 position_j = FETCH_VAR(var_sorted, index_j, DENS);
		//float4 position_j = FETCH_VAR(var_sorted, index_j, DENS);
		float4 position_j = FETCH_VAR(var_sorted, index_j, 0); // uses numParticles

		// get the relative distance between the two particles, translate to simulation space
		float scale_to_simulation = 1.0;
		float4 r = (position_i - position_j) * scale_to_simulation;
		//float3 r = (position_i - position_j) * cFluidParams.scale_to_simulation;

		float rlen_sq = dot(r,r);
		// |r|
		float rlen = sqrtf(rlen_sq);

		rlen = sqrt(rlen_sq);

		// is this particle within cutoff?
		float smoothing_length = 1.0;
		if (rlen <= smoothing_length){
#if 1
			ForNeighbor(var_sorted, index_i, index_j, r, rlen, rlen_sq);
#endif
		}
	}
}

//--------------------------------------------------
#endif
