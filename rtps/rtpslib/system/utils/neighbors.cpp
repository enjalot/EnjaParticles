#ifndef __NEIGHBORS_CL_K_
#define __NEIGHBORS_CL_K_

#include "cl_macros.h"
#include "cl_structures.h"
#include "wpoly6.cl"

//----------------------------------------------------------------------
float4 ForNeighbor(__global float4*  vars_sorted,
				__constant uint index_i,
				uint index_j,
				float4 r,
				float rlen,
				float rlen_sq,
	  			__constant struct GridParams* gp,
	  			__constant struct FluidParams* fp,
	  			__constant struct SPHParams* sphp)
{
// the density sum using Wpoly6 kernel
// data.sum_density += SPH_Kernels::Wpoly6::Kernel_Variable(fp->smoothing_length_pow2, r, rlen_sq);	
// #include FILE to deal with collisions or other stuff

	if (fp->choice == 1) {
		// update density
		// return density.x for single neighbor
		#include "density_update.cl"
		;
	}

	if (fp->choice == 2) {
		// update pressure
		//#include "pressure_update.cl"
	}


//	int index = get_global_id(0);

//#include "cl_snippet_sphere_forces.h"
	//return r;
}
//--------------------------------------------------
float4 ForPossibleNeighbor(__global float4* vars_sorted, 
						__constant uint numParticles, 
						__constant uint index_i, 
						uint index_j, 
						__constant float4 position_i,
	  					__constant struct GridParams* gp,
	  					__constant struct FluidParams* fp,
	  					__constant struct SPHParams* sphp)
{
	float4 frce;
	frce.x = 0.;
	frce.y = 0.;
	frce.z = 0.;
	frce.w = 0.;
	//float4 force = convert_float4(0.);

	// check not colliding with self
	if (index_j != index_i) {
		// get the particle info (in the current grid) to test against
		float4 position_j = FETCH_POS(vars_sorted, index_j); // uses numParticles

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j) * fp->scale_to_simulation;

		float rlen_sq = dot(r,r);
		// |r|
		float rlen;
		rlen = sqrt(rlen_sq);

		// is this particle within cutoff?
		if (rlen <= fp->smoothing_length) {
#if 1
			frce = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, rlen_sq, gp, fp, sphp);
#endif
		}
	}
	return frce;
}
//--------------------------------------------------
#endif
