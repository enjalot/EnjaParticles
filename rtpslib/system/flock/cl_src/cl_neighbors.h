#ifndef _CL_FLOCK_NEIGHBORS_H_
#define _CL_FLOCK_NEIGHBORS_H_

#include "cl_hash.h"

//----------------------------------------------------------------------
void zeroPoint(Boid* pt)
{
	pt->separation= (float4)(0.,0.,0.,0.);
	pt->alignment= (float4)(0.,0.,0.,0.);
	pt->cohesion= (float4)(0.,0.,0.,0.);
	pt->acceleration= (float4)(0.,0.,0.,0.);
	pt->color= (float4)(0.,0.,0.,0.);
	pt->num_flockmates = 0;
    pt->num_nearestFlockmates = 0;
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i)
*/
void IterateParticlesInCell(
                            // __global float4*    vars_sorted,
                            ARGS,
                            Boid* pt,
                            uint num,
                            int4 	cellPos,
                            uint 	index_i,
                            float4 	position_i,
                            __global int* 		cell_indexes_start,
                            __global int* 		cell_indexes_end,
                            __constant struct GridParams* gp,
                            __constant struct FLOCKParameters* flockp
                            DEBUG_ARGS
)
{
	//clf[index] = position_i; 
	//clf[index].w = -128.;
	//return;
    // get hash (of position) of current cell
    uint cellHash = calcGridHash(cellPos, gp->grid_res, false);

    //need to check cellHash to make sure its not out of bounds
    if(cellHash >= gp->nb_cells)
    {
        return;
    }
    
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
            //ForPossibleNeighbor(vars_sorted, pt, num, index_i, index_j, position_i, gp, /*fp,*/ flockp DEBUG_ARGV);
            ForNeighbor(/*vars_sorted*/ARGV, pt, index_i, index_j, position_i, gp, /*fp,*/ flockp DEBUG_ARGV);
#endif
        }
        //clf[index_i] = pt->force;
    }
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i) 
 */
void IterateParticlesInNearbyCells(
                                //__global float4* vars_sorted,
                                ARGS,
                                Boid* pt,
                                uint num,
                                int 	index_i, 
                                float4   position_i, 
                                __global int* 		cell_indices_start,
                                __global int* 		cell_indices_end,
                                __constant struct GridParams* gp,
                                __constant struct FLOCKParameters* flockp
                                DEBUG_ARGS
                                )       
{
	//clf[index_i] = position_i; 
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
                IterateParticlesInCell(/*vars_sorted*/ARGV, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ flockp DEBUG_ARGV);

            //barrier(CLK_LOCAL_MEM_FENCE); // DEBUG
            // SERIOUS PROBLEM: Results different than results with cli = 5 (bottom of this file)
            }
        }
    }
}

//----------------------------------------------------------------------
 

#if 0
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

        //clf[index_i].x = rlen;

        //This shouldn't be happening!?
        //two particles shouldn't be able to be in the same place at same time
        //must be some other bug?
		// never compare a float against zero!!! Check that rlen < eps, where 
		// eps is very small
        if (rlen == 0.0 && flockp->choice != 0) return;

		// is this particle within cutoff?

		if (rlen <= flockp->smoothing_distance) {
#if 1
            //cli[index_i].w += 1;
			// return updated pt
			ForNeighbor(vars_sorted, pt, index_i, index_j, r, rlen, gp,/* fp,*/ flockp /*DEBUG_ARGV*/);
#endif
		}
	}
}
#endif
//--------------------------------------------------
#endif
