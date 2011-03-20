#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


/* TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 
*/

/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"

//Contains all of the Smoothing Kernels for FLOCK
#include "cl_kernels.h"

//----------------------------------------------------------------------
inline void ForNeighbor(__global float4*  vars_sorted,
				PointData* pt,
				uint index_i,
				uint index_j,
				float4 position_i,
	  			__constant struct GridParams* gp,
	  			__constant struct FLOCKParams* flockp
                DEBUG_ARGS
				)
{
    int num = flockp->num;
	
	// get the particle info (in the current grid) to test against
	float4 position_j = pos(index_j); 

	float4 r = (position_i - position_j); 
	r.w = 0.f; 
	
    // |r|
	float rlen = length(r);

    // parameter that would be moved to FLOCKparams	
	float searchradius = 0.8f;  //8.f; 	    // search radius TODO: remove hard coded parameter

	//clf[index_i] = position_i; 
	//clf[index_i].w = -129.;
	//return;

	//clf[index_i].x = rlen;
	//clf[index_i].y = searchradius;
	//clf[index_i] = r;
	//clf[index_i].w = -125.;
	//return;

	//clf[index_i] = pos(index_i);
	clf[index_i].z = flockp->min_dist;
	clf[index_i].w = -123.;
	return;

    // is this particle within cutoff?
	clf[index_i] = position_i; 
	if (rlen <= searchradius) 
    {
		//clf[index_i].x++;
		//clf[index_i].x = 13.;
		//clf[index_i].y += rlen;
		//clf[index_i].z = searchradius;
		//clf[index_i].w = flockp->smoothing_distance;
		//return;

        if (flockp->choice == 0) {
            // are the boids the same? 
            int iej = index_i != index_j;

            // compute the rules 
            #include "cl_density.h"
	// searchradius = 0.8
			//clf[index_i].w = searchradius;
	return;
        }

        if (flockp->choice == 1) {
            //iej is 0 when we are looking at same particle
            //we allow calculations and just multiply force and xflock
            //by iej to avoid branching
            int iej = index_i != index_j;
                
            // update pressure
            //#include "cl_force.h"
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
				__constant struct FLOCKParams* flockp 
				DEBUG_ARGS
				)
{
    // particle index
	int nb_vars = flockp->nb_vars;
	int num = flockp->num;

    int index = get_global_id(0);
    if (index >= num) return;

	clf[index] = (float4)(0.,0.,0.,10.);
	cli[index] = (int4)(0.,0.,0.,0.);

    float4 position_i = pos(index);

    //debuging
    //clf[index] = (float4)(0,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
	PointData pt;
	zeroPoint(&pt);

	if (flockp->choice == 0) { // update density
	//clf[index] = position_i; 
	//clf[index].w = -127.;
	//return;

    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
        
        //pt.density = (float4)(5., 5., 5., 5.);
        //pt.force= (float4)(1., 1., 1., 1.);
		
        den(index) = pt.density;
		xflock(index) = pt.xflock;
        force(index) = pt.force;
        surface(index) = pt.surf_tens;

        //clf[index].xyz= force(index).xyz;
        //clf[index].w = den(index).x;
	}
#if 0
	if (flockp->choice == 1) { // update force
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		force(index) = pt.force; // Does not seem to maintain value into euler.cl
        //clf[index].xyz = pt.force.xyz;
	}

	if (flockp->choice == 2) { // update surface tension (NOT DEBUGGED)
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ flockp DEBUG_ARGV);
		float norml = length(pt.color_normal);
		if (norml > 1.) {
			float4 stension = -0.3f * pt.color_lapl * pt.color_normal / norml;
			force(index) += stension; // 2 memory accesses (NOT GOOD)
		}
	}
	if (flockp->choice == 3) { // denominator in density normalization
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ flockp DEBUG_ARGV);

		density(index) /= pt.density.y;
	}
	
	/*if (flockp->choice == 4) { //Extract surface particles
		IterateParticlesInNearbyCells(vars_sorted,&pt,num,index, position_i, cell_indexes_start, cell_indexes_end, gp, flockp DEBUG_ARGV);
		
		pt.center_of_mass = pt.center_of_mass/(float) pt.num_neighbors;
		float4 dist = pos(index)-pt.center_of_mass;
		dist.w = 0;
		if(pt.num_neighbors < 5 ||
			sqrt(dot(dist,dist)) > flockp->surface_threshold)	
			surface(index) = (float4){1.0,1.0,1.0,1.0};
		else
			surface(index) = (float4){0.0,0.0,0.0,0.0};
	}*/
#endif
}

/*-------------------------------------------------------------- */
#endif

