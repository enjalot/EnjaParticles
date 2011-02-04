#ifndef __NEIGHBORS_CL_K_
#define __NEIGHBORS_CL_K_

#include "cl_kernels.h"

//----------------------------------------------------------------------
void zeroPoint(PointData* pt)
{
	pt->density = (float4)(0.,0.,0.,0.);
	pt->color = (float4)(0.,0.,0.,0.);
	pt->color_normal = (float4)(0.,0.,0.,0.);
	pt->force = (float4)(0.,0.,0.,0.);
	pt->surf_tens = (float4)(0.,0.,0.,0.);
	pt->color_lapl = 0.;
	pt->xflock = (float4)(0.,0.,0.,0.);
//	pt->center_of_mass = (float4)(0.,0.,0.,0.);
//	pt->num_neighbors = 0;
}
//----------------------------------------------------------------------
inline void ForNeighbor(__global float4*  vars_sorted,
				PointData* pt,
				uint index_i,
				uint index_j,
				float4 r,
				float rlen,
	  			__constant struct GridParams* gp,
	  			//__constant struct FluidParams* fp,
	  			__constant struct FLOCKParams* flockp
				)
{
    int num = flockp->num;
    //int num = get_global_size(0);


	if (flockp->choice == 0) {
		// update density
		// return density.x for single neighbor
		#include "cl_density.h"
	}

	if (flockp->choice == 1) {
		// update pressure
		#include "cl_force.h"
	}

	if (flockp->choice == 2) {
		// update color normal and color Laplacian
		//#include "cl_surface_tension.h"
	}

	if (flockp->choice == 3) {
		//#include "density_denom_update.cl"
	} 
/*	
	if (flockp->choice == 4) {
		#include "cl_surface_extraction.h"
	}*/
}
//--------------------------------------------------
inline void ForPossibleNeighbor(__global float4* vars_sorted, 
						PointData* pt,
                        uint numParticles,
						uint index_i, 
						uint index_j, 
						float4 position_i,
	  					__constant struct GridParams* gp,
	  					//__constant struct FluidParams* fp,
	  					__constant struct FLOCKParams* flockp
	  					DEBUG_ARGS
						)
{
	// self-collisions ok when computing density
	// no self-collisions in the case of pressure

	if (flockp->choice == 0 || (index_j != index_i)) {  // RESTORE WHEN DEBUGGED
	//{
        //int num = get_global_size(0); //this was being passed through all the functions; could but put back..
		// get the particle info (in the current grid) to test against
		float4 position_j = pos(index_j); 

		// get the relative distance between the two particles, translate to simulation space
		float4 r = (position_i - position_j); 
		r.w = 0.f; // I stored density in 4th component
		// |r|
		float rlen = length(r);

		// is this particle within cutoff?

		if (rlen <= flockp->smoothing_distance) {
#if 1
            cli[index_i].w += 1;
			// return updated pt
			ForNeighbor(vars_sorted, pt, index_i, index_j, r, rlen, gp,/* fp,*/ flockp /*DEBUG_ARGV*/);
#endif
		}
	}
}
//--------------------------------------------------
#endif
