/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


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
