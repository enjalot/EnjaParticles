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
	  			__constant struct GridParams* gp,
	  			__constant struct FluidParams* fp,
	  			__constant struct SPHParams* sphp
	  			DUMMY_ARGS
				)
{
	int num = get_global_size(0);

	//cli[index_i].x++;
	//cli[index_i].y = fp->choice;
	//clf[index_i].x = fp->choice;
	//cli[index_i].x = fp->choice;

	if (fp->choice == 0) {
		cli[index_i].y++;
		//cli[index_i].w = -999.;
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
	// no self-collisions in the case of pressure
	if (fp->choice == 0 || index_j != index_i) {  // RESTORE WHEN DEBUGGED
	//{
		// get the particle info (in the current grid) to test against
		float4 position_j = pos(index_j); 

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j); 
		// |r|
		float rlen = length(r);

		// is this particle within cutoff?

		if (rlen <= sphp->smoothing_distance) {
			//cli[index_i].x++;
#if 1
			frce = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, gp, fp, sphp ARGS);
#endif
		}
	}
	return frce;
}
//--------------------------------------------------
#endif
