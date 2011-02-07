#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


/* TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 
*/

/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"

//Contains all of the Smoothing Kernels for SPH
#include "cl_kernels.h"

   

//----------------------------------------------------------------------
inline void ForNeighbor(__global float4*  vars_sorted,
				PointData* pt,
				uint index_i,
				uint index_j,
				float4 position_i,
	  			__constant struct GridParams* gp,
	  			__constant struct SPHParams* sphp
                DEBUG_ARGS
				)
{
    int num = sphp->num;
	
    //if (sphp->choice == 0 || (index_j != index_i)) 
    //{
		// get the particle info (in the current grid) to test against
		float4 position_j = pos(index_j); 

		float4 r = (position_i - position_j); 
		r.w = 0.f; // I stored density in 4th component
		// |r|
		float rlen = length(r);

		// is this particle within cutoff?
		if (rlen <= sphp->smoothing_distance) 
        {

            if (sphp->choice == 0) {
                // update density
                // return density.x for single neighbor
                #include "cl_density.h"
            }

            if (sphp->choice == 1) {

                //iej is 0 when we are looking at same particle
                //we allow calculations and just multiply force and xsph
                //by iej to avoid branching
                int iej = index_i != index_j;
                
                
                // update pressure
                #include "cl_force.h"
            }

            if (sphp->choice == 2) {
                // update color normal and color Laplacian
                //#include "cl_surface_tension.h"
            }

            if (sphp->choice == 3) {
                //#include "density_denom_update.cl"
            } 
        /*	
            if (sphp->choice == 4) {
                #include "cl_surface_extraction.h"
            }*/
        }
    //}
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// compute forces on particles

__kernel void neighbors(
				__global float4* vars_sorted,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
				__constant struct GridParams* gp,
				__constant struct SPHParams* sphp 
				DEBUG_ARGS
				)
{
    // particle index
	int nb_vars = sphp->nb_vars;
	int num = sphp->num;
    //int numParticles = get_global_size(0);
    //int num = get_global_size(0);

	int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos(index);

    //debuging
    //clf[index] = (float4)(0,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
	PointData pt;
	zeroPoint(&pt);

	if (sphp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
		density(index) = sphp->wpoly6_coef * pt.density.x;
        //clf[index].w = density(index);
	}
	if (sphp->choice == 1) { // update force
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
		force(index) = pt.force; // Does not seem to maintain value into euler.cl
        //clf[index].xyz = pt.force.xyz;
		xsph(index) = sphp->wpoly6_coef * pt.xsph;
	}
#if 0
	if (sphp->choice == 2) { // update surface tension (NOT DEBUGGED)
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ sphp DEBUG_ARGV);
		float norml = length(pt.color_normal);
		if (norml > 1.) {
			float4 stension = -0.3f * pt.color_lapl * pt.color_normal / norml;
			force(index) += stension; // 2 memory accesses (NOT GOOD)
		}
	}
	if (sphp->choice == 3) { // denominator in density normalization
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ sphp DEBUG_ARGV);

		density(index) /= pt.density.y;
	}
	
	/*if (sphp->choice == 4) { //Extract surface particles
		IterateParticlesInNearbyCells(vars_sorted,&pt,num,index, position_i, cell_indexes_start, cell_indexes_end, gp, sphp DEBUG_ARGV);
		
		pt.center_of_mass = pt.center_of_mass/(float) pt.num_neighbors;
		float4 dist = pos(index)-pt.center_of_mass;
		dist.w = 0;
		if(pt.num_neighbors < 5 ||
			sqrt(dot(dist,dist)) > sphp->surface_threshold)	
			surface(index) = (float4){1.0,1.0,1.0,1.0};
		else
			surface(index) = (float4){0.0,0.0,0.0,0.0};
	}*/
#endif
}

/*-------------------------------------------------------------- */
#endif

