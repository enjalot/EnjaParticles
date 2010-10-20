#ifndef __NEIGHBORS_CL_K_
#define __NEIGHBORS_CL_K_

#include "cl_macros.h"
#include "cl_structures.h"
#include "wpoly6.cl"

//----------------------------------------------------------------------
void zeroPoint(PointData* pt)
{
	pt->density = (float4)(0.,0.,0.,0.);
	pt->color = (float4)(0.,0.,0.,0.);
	pt->color_normal = (float4)(0.,0.,0.,0.);
	pt->force = (float4)(0.,0.,0.,0.);
	pt->surf_tens = (float4)(0.,0.,0.,0.);
	pt->color_lapl = 0.;
	pt->xsph = (float4)(0.,0.,0.,0.);
}
//----------------------------------------------------------------------
void ForNeighbor(__global float4*  vars_sorted,
				PointData* pt,
				uint index_i,
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


	if (fp->choice == 0) {
		// update density
		// return density.x for single neighbor
		#include "density_update.cl"
	}

	if (fp->choice == 1) {
		// update pressure
		#include "pressure_update.cl"
	}

	if (fp->choice == 2) {
		// update color normal and color Laplacian
		#include "surface_tension_update.cl"
	}

	if (fp->choice == 3) {
		//#include "density_denom_update.cl"
	} 
}
//--------------------------------------------------
void ForPossibleNeighbor(__global float4* vars_sorted, 
						PointData* pt,
						uint numParticles, 
						uint index_i, 
						uint index_j, 
						float4 position_i,
	  					__constant struct GridParams* gp,
	  					__constant struct FluidParams* fp,
	  					__constant struct SPHParams* sphp
	  					DUMMY_ARGS
						)
{
	// not really needed if pt approach works

	// check not colliding with self
	//if (index_j != index_i) {  // RESTORE WHEN DEBUGGED

	// self-collisions ok when computing density
	// no self-collisions in the case of pressure


	if (fp->choice == 0 || (index_j != index_i)) {  // RESTORE WHEN DEBUGGED
	//{
		// get the particle info (in the current grid) to test against
		float4 position_j = pos(index_j); 

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j); 
		r.w = 0.f; // I stored density in 4th component
		// |r|
		float rlen = length(r);

		// is this particle within cutoff?

		if (rlen <= sphp->smoothing_distance) {
#if 1
			// return updated pt
			ForNeighbor(vars_sorted, pt, index_i, index_j, r, rlen, gp, fp, sphp ARGS);
#endif
		}
	}
}
//--------------------------------------------------
#endif
