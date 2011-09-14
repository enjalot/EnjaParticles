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


#ifndef __NEIGHBORS_CL_K_
#define __NEIGHBORS_CL_K_

#include "cl_hash.h"

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
    //	pt->center_of_mass = (float4)(0.,0.,0.,0.);
    //	pt->num_neighbors = 0;
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i)
*/
void IterateParticlesInCell(
                           //__global float4*    vars_sorted,
                           ARGS,
                           PointData* pt,
                           uint num,
                           int4    cellPos,
                           uint    index_i,
                           float4  position_i,
                           __global int*       cell_indexes_start,
                           __global int*       cell_indexes_end,
                           __constant struct GridParams* gp,
                           __constant struct SPHParams* sphp
                           DEBUG_ARGS
                           )
{
    // get hash (of position) of current cell
    uint cellHash = calcGridHash(cellPos, gp->grid_res, false);
    
    //need to check cellHash to make sure its not out of bounds
    if(cellHash >= gp->nb_cells)
    {
        return;
    }
    //even with cellHash in bounds we are still getting out of bounds indices

    /* get start/end positions for this cell/bucket */
    uint startIndex = FETCH(cell_indexes_start,cellHash);
    /* check cell is not empty
     * WHERE IS 0xffffffff SET?  NO IDEA ************************
     */
    if (startIndex != 0xffffffff)
    {
        uint endIndex = FETCH(cell_indexes_end, cellHash);

        /* iterate over particles in this cell */
        for (uint index_j=startIndex; index_j < endIndex; index_j++)
        {
#if 1
            //***** UPDATE pt (sum)
            //ForPossibleNeighbor(vars_sorted, pt, num, index_i, index_j, position_i, gp, /*fp,*/ sphp DEBUG_ARGV);
            //ForNeighbor(vars_sorted, pt, index_i, index_j, position_i, gp, sphp DEBUG_ARGV);
            ForNeighbor(ARGV, pt, index_i, index_j, position_i, gp, sphp DEBUG_ARGV);
#endif
        }
    }
}

/*--------------------------------------------------------------*/
/* Iterate over particles found in the nearby cells (including cell of position_i) 
 */
void IterateParticlesInNearbyCells(
                                  //__global float4* vars_sorted,
                                  ARGS,
                                  PointData* pt,
                                  uint num,
                                  int     index_i, 
                                  float4   position_i, 
                                  __global int*       cell_indices_start,
                                  __global int*       cell_indices_end,
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
    //TODO bug in that we don't take into account cells on edge of grid!
    for (int z=cell.z-1; z<=cell.z+1; ++z)
    {
        for (int y=cell.y-1; y<=cell.y+1; ++y)
        {
            for (int x=cell.x-1; x<=cell.x+1; ++x)
            {
                int4 ipos = (int4) (x,y,z,1);

                // **** SUMMATION/UPDATE
                //IterateParticlesInCell(vars_sorted, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ sphp DEBUG_ARGV);
                IterateParticlesInCell(ARGV, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ sphp DEBUG_ARGV);

                //barrier(CLK_LOCAL_MEM_FENCE); // DEBUG
                // SERIOUS PROBLEM: Results different than results with cli = 5 (bottom of this file)
            }
        }
    }
}

//----------------------------------------------------------------------

#endif
