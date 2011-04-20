#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float* density, __global float4* color
#define ARGV pos, density, color


/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"
//Contains all of the Smoothing Kernels for SPH
#include "cl_kernels.h"


//----------------------------------------------------------------------
inline void ForNeighbor(//__global float4*  vars_sorted,
                        ARGS,
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

    // get the particle info (in the current grid) to test against
    float4 position_j = pos[index_j] * sphp->simulation_scale; 
    float4 colj = color[index_j];
    colj.y = 0.f;
    float densj = density[index_j] + sphp->EPSILON;
    float4 r = (position_i - position_j); 
    r.w = 0.f; // I stored density in 4th component
    // |r|
    float rlen = length(r);
    //pt->density.y += 1.;

    int iej = index_i != index_j;
    // is this particle within cutoff?
    if (rlen <= sphp->smoothing_distance)
    {
        // return density.x for single neighbor
        //float Wij = sphp->wpoly6_coef * Wpoly6(r, sphp->smoothing_distance, sphp);
        //float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);
        float dWijlapl = sphp->wvisc_dd_coef * Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
        //float dWijlapl = Wpoly6_lapl(rlen, sphp->smoothing_distance, sphp);
        //pt->density.x += sphp->mass*Wij;
        //diffusioin coefficient needs to be moved to parameters
        float diffuse_coeff = .00000001f;
        pt->color += diffuse_coeff * sphp->mass * colj * dWijlapl / densj; 
        pt->color = pt->color * (float)iej;
    }
}
//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"
//--------------------------------------------------------------
// compute forces on particles

__kernel void diffuse(
//                       __global float4* vars_sorted,
                       ARGS,
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

#if 1
    float4 position_i = pos[index] * sphp->simulation_scale;

    //debuging
    //clf[index] = (float4)(99,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

    //IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    IterateParticlesInNearbyCells(ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    color[index] = pt.color * density[index];
    color[index].y = 0.f;
    /*
    clf[index].x = pt.density.x * sphp->wpoly6_coef;
    clf[index].y = pt.density.y;
    clf[index].z = sphp->smoothing_distance;
    clf[index].w = sphp->mass;
    */

    //int4 cell = calcGridCell(pos[index], gp->grid_min, gp->grid_delta);
    /*
    int4 cell = calcGridCell(pos[index] * sphp->simulation_scale, gp->grid_min, gp->grid_delta);
    uint cellHash = calcGridHash(cell, gp->grid_res, false);
    clf[index].x = cellHash;
    clf[index].y = pt.density.y;
    clf[index].z = pt.density.z;
    clf[index].w = pt.density.w;
    */
#endif
}

/*-------------------------------------------------------------- */
#endif

