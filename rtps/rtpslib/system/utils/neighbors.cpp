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
	  			__constant struct SPHParams* sphp
	  			DUMMY_ARGS
				)
{
	int num = get_global_size(0);

	cli[index_i].x++;

	if (fp->choice == 0) {
		// update density
		// return density.x for single neighbor
		#include "density_update.cl"
	}

	if (fp->choice == 1) {
		// update pressure
		#include "pressure_update.cl"
	}
}
//--------------------------------------------------
float4 ForPossibleNeighbor(__global float4* vars_sorted, 
						__constant uint numParticles, 
						__constant uint index_i, 
						uint index_j, 
						__constant float4 position_i,
	  					__constant struct GridParams* gp,
	  					__constant struct FluidParams* fp,
	  					__constant struct SPHParams* sphp
	  					DUMMY_ARGS
						)
{
	float4 frce = (float4) (0.,0.,0.,0.);

	// check not colliding with self
	//if (index_j != index_i) {  // RESTORE WHEN DEBUGGED

	// self-collisions ok when computing density
	//if (fp->choice == 0 || index_j != index_i) {  // RESTORE WHEN DEBUGGED
		//cli[index_i].y++;    // two less than cli[index_i].x  (WHY???) Should be one less
	{
		// get the particle info (in the current grid) to test against
		float4 position_j = FETCH_POS(vars_sorted, index_j); // uses numParticles

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j); // * fp->scale_to_simulation;

		float rlen_sq = dot(r,r);
		// |r|
		float rlen = length(r);
		//clf[index_i].x = rlen;
		//clf[index_i].w = sphp->smoothing_distance;

		// is this particle within cutoff?
	clf[index_i].x =  rlen;
	clf[index_i].y =  sphp->smoothing_distance;
	return frce;

		if (rlen <= sphp->smoothing_distance) {
			cli[index_i].z++;
#if 1
			frce = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, rlen_sq, gp, fp, sphp ARGS);
#endif
		}
	}
	return frce;
}
//--------------------------------------------------
#endif
