#ifndef _CL_FLOCK_NEIGHBORS_H_
#define _CL_FLOCK_NEIGHBORS_H_

#include "cl_hash.h"

//----------------------------------------------------------------------
void zeroPoint(Boid* pt)
{
	pt->separation  = (float4)(0.,0.,0.,0.);
	pt->alignment   = (float4)(0.,0.,0.,0.);
	pt->cohesion    = (float4)(0.,0.,0.,0.);
	pt->goal        = (float4)(0.,0.,0.,0.);
	pt->avoid       = (float4)(0.,0.,0.,0.);
	pt->leaderfollowing= (float4)(0.,0.,0.,0.);
	pt->color       = (float4)(0.,0.,0.,0.);
	pt->num_flockmates = 0;
    pt->num_nearestFlockmates = 0;
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i)
*/
void IterateParticlesInCell(ARGS,
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
    // get hash (of position) of current cell
    uint cellHash = calcGridHash(cellPos, gp->grid_res, false);

    //need to check cellHash to make sure its not out of bounds
    if(cellHash >= gp->nb_cells){ return; }
    
    /* get start/end positions for this cell/bucket */
    uint startIndex = FETCH(cell_indexes_start,cellHash);
    
    /* check cell is not empty */
    if (startIndex != 0xffffffff) {	   
        uint endIndex = FETCH(cell_indexes_end, cellHash);

        /* iterate over particles in this cell */
        for(uint index_j=startIndex; index_j < endIndex; index_j++) {			
            ForNeighbor(ARGV, pt, index_i, index_j, position_i, gp, flockp DEBUG_ARGV);
        }
    }
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i) 
 */
void IterateParticlesInNearbyCells(ARGS,
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
    int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);

    // iterate through the 3^3 cells in and around the given position
    // can't unroll these loops, they are not innermost 
    for(int z=cell.z-1; z<=cell.z+1; ++z) {
        for(int y=cell.y-1; y<=cell.y+1; ++y) {
            for(int x=cell.x-1; x<=cell.x+1; ++x) {
                int4 ipos = (int4) (x,y,z,1);
                IterateParticlesInCell(ARGV, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp, flockp DEBUG_ARGV);
            }
        }
    }
}

#endif
