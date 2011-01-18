#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


/* TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 
*/

/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"
#include "cl_neighbors.h"
#include "cl_hash.h"


	/*--------------------------------------------------------------*/
	/* Iterate over particles found in the nearby cells (including cell of position_i)
	*/
	void IterateParticlesInCell(
		__global float4*    vars_sorted,
		PointData* pt,
        uint num,
		int4 	cellPos,
		uint 	index_i,
		float4 	position_i,
		__global int* 		cell_indexes_start,
		__global int* 		cell_indexes_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct SPHParams* sphp
		DEBUG_ARGS
    )
	{
		// get hash (of position) of current cell
		uint cellHash = calcGridHash(cellPos, gp->grid_res, false);

		/* get start/end positions for this cell/bucket */
		uint startIndex = FETCH(cell_indexes_start,cellHash);
		/* check cell is not empty
		 * WHERE IS 0xffffffff SET?  NO IDEA ************************
		 */
		if (startIndex != 0xffffffff) {	   
			uint endIndex = FETCH(cell_indexes_end, cellHash);

			/* iterate over particles in this cell */
			for(uint index_j=startIndex; index_j < endIndex; index_j++) {			
#if 1
				//***** UPDATE pt (sum)
				ForPossibleNeighbor(vars_sorted, pt, num, index_i, index_j, position_i, gp, /*fp,*/ sphp DEBUG_ARGV);
#endif
			}
		}
	}

	/*--------------------------------------------------------------*/
	/* Iterate over particles found in the nearby cells (including cell of position_i) 
	 */
	void IterateParticlesInNearbyCells(
		__global float4* vars_sorted,
		PointData* pt,
        uint num,
		int 	index_i, 
		float4   position_i, 
		__global int* 		cell_indices_start,
		__global int* 		cell_indices_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct SPHParams* sphp
		DEBUG_ARGS
		)
	{
		// initialize force on particle (collisions)

		// get cell in grid for the given position
		//int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_inv_delta);
		int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost 
		for(int z=cell.z-1; z<=cell.z+1; ++z) {
			for(int y=cell.y-1; y<=cell.y+1; ++y) {
				for(int x=cell.x-1; x<=cell.x+1; ++x) {
					int4 ipos = (int4) (x,y,z,1);

					// **** SUMMATION/UPDATE
					IterateParticlesInCell(vars_sorted, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ sphp DEBUG_ARGV);

				//barrier(CLK_LOCAL_MEM_FENCE); // DEBUG
				// SERIOUS PROBLEM: Results different than results with cli = 5 (bottom of this file)
				}
			}
		}
	}

	//----------------------------------------------------------------------
//--------------------------------------------------------------
// compute forces on particles

void IterateGhosts(
		__global float4* vars_sorted,
		__global float4* ghosts,
		PointData* pt,
        uint num,
		uint 	index_i,
		float4   position_i, 
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct SPHParams* sphp
		DEBUG_ARGS
		)
{
    int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);

    float4 di = density(index_i);  // should not repeat

    float4 gdeb = ghosts[calcGridHash(cell, gp->grid_res, false)];
    
    /*
    cli[index_i].x = (int)(position_i.x*10000);
    cli[index_i].y = (int)(position_i.y*10000);
    cli[index_i].z = (int)(position_i.z*10000);
    */
    cli[index_i] = cell;
    cli[index_i].w = calcGridHash(cell, gp->grid_res, false);
    
    clf[index_i].xy = gdeb.xy;//*sphp->simulation_scale;
    clf[index_i].zw = position_i.xy;
    clf[index_i].x = 0;
    clf[index_i].y = 0;
    //cli[index_i] = cell;


    // iterate through the 3^3 cells in and around the given position
    // can't unroll these loops, they are not innermost 
    for(int z=cell.z-1; z<=cell.z+1; ++z) 
    {
        for(int y=cell.y-1; y<=cell.y+1; ++y) 
        {
            for(int x=cell.x-1; x<=cell.x+1; ++x) 
            {
                int4 ipos = (int4) (x,y,z,1);
		        uint cellHash = calcGridHash(ipos, gp->grid_res, false);
                

                float4 gpos = ghosts[cellHash];// * sphp->simulation_scale;//one ghost per grid cell
                // get the relative distance between the two particles, translate to simulation space
                float4 r = (position_i - gpos); 
                r.w = 0.f; // I stored density in 4th component
                // |r|
                float rlen = length(r);
                clf[index_i].x = rlen;
                clf[index_i].y = sphp->smoothing_distance;
                clf[index_i].z = di.x;
                clf[index_i].w = 0;
                if(rlen < sphp->smoothing_distance)
                {
                    //rlen = rlen / 2.;
                    if(sphp->choice == 0)
                    {
                        //calculate density from ghost
                        float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);
                        pt->density.x += 3.0f * (1.f - gpos.w) * sphp->mass*Wij;

                    }
                    else if(sphp->choice == 1)
                    {
                        float casper = gpos.w/sphp->simulation_scale;
                        //calculate force from ghost
                        // gradient
                        float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

                        float gdense = 1000.f * (1.7f - casper);
                        float dj = gdense + 400 * (1.f - casper);
                        //float dj = gdense * (1.7f - casper);
                        //float dj = 1000.;

                        //form simple SPH in Krog's thesis

                        float rest_density = 1000.f;
                        float Pi = sphp->K*(di.x - rest_density);
                        //float Pj = sphp->K*(dj - rest_density);
                        float Pj = sphp->K*(dj - gdense);

                        float kern = -dWijdr * (Pi + Pj)*0.5f * sphp->wspiky_d_coef;
                        float4 stress = kern*r; // correct version

                        ///*
                        float4 veli = veleval(index_i); // sorted
                        float4 velj = -veli;

                        // Add viscous forces

                        #if 0
                        //float vvisc = 1.0f;
                        float visc = 1.01f - casper;
                        float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
                        stress += visc * (velj-veli) * dWijlapl;
                        #endif
                        //*/
                        //stress *=  sphp->mass/(di.x*dj);  // original
                        float mj = sphp->mass * (1.5f - casper);
                        stress *=  mj/(di.x*dj);
                        pt->force += stress;

                        float Wijpol6 = Wpoly6(r, sphp->smoothing_distance, sphp);
	                    //pt->xsph +=  (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di.x+dj))*(1.3f-casper);
	                    pt->xsph +=  ((sphp->mass + mj) * Wijpol6 * (velj-veli)/(di.x+dj));
                        //pt->force += (float4)(0,0,-.1,0);
                        clf[index_i] = stress;
                    }
                }
            }
        }
    }
}


__kernel void neighbors(
				__global float4* vars_sorted,
                __global float4* ghosts,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
				__constant struct GridParams* gp,
				//__constant struct FluidParams* fp, 
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
    //cli[index].w = 0;


    // Do calculations on particles in neighboring cells
	PointData pt;
	zeroPoint(&pt);

	if (sphp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
		density(index) = sphp->wpoly6_coef * pt.density.x;
        
        //IterateGhosts(vars_sorted, ghosts, &pt, num, index, position_i, gp, sphp DEBUG_ARGV);

        //clf[index].w = density(index);
		// code reaches this point on first call
	}
	if (sphp->choice == 1) { // update force
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
        
        IterateGhosts(vars_sorted, ghosts, &pt, num, index, position_i, gp, sphp DEBUG_ARGV);
		
        force(index) = pt.force; // Does not seem to maintain value into euler.cl
        

        //clf[index].xyz = pt.force.xyz;
		xsph(index) = sphp->wpoly6_coef * pt.xsph;
	}

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
}

/*-------------------------------------------------------------- */
#endif

