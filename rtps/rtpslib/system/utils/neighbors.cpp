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
//clf[index_i].w = -17.;
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
		//cli[index_i].y++;    // two less than cli[index_i].x  (WHY???) Should be one less
	//{
		// get the particle info (in the current grid) to test against
		float4 position_j = pos(index_j); //FETCH_POS(vars_sorted, index_j); // uses numParticles

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j); 
	//clf[index_i] = position_i; return frce;
		float rlen_sq = dot(r,r);
		// |r|
		float rlen = length(r);
		//clf[index_i].x = rlen;
		//clf[index_i].z = rlen;
		//clf[index_i].w = sphp->smoothing_distance;
		cli[index_i].x++;
		//return frce;

		// is this particle within cutoff?
	//clf[index_i].x =  rlen;
	//clf[index_i].y =  sphp->smoothing_distance;
	//return frce;

		if (rlen <= sphp->smoothing_distance) {
			//cli[index_i].z++;
#if 1
			frce = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, rlen_sq, gp, fp, sphp ARGS);
			// the force summation is different than if I put clf inside pressure_update.cl
			//clf[index_i] += frce;
			//cli[index_i].w = 2;
#endif
		}
	}
	return frce;
}
//--------------------------------------------------
#endif
