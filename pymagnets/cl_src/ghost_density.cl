#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float4* ghost_pos, __global float* density, __global float* ghost_density, __global float4* ghost_intensity, __constant struct SPHParams* particle_sphp, float target_intensity
#define ARGV pos, ghost_pos, density, ghost_density, ghost_intensity, particle_sphp, target_intensity


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

    // get the particle info (in the current grid) to test against
    float4 position_j = ghost_pos[index_j] * sphp->simulation_scale; 
    float4 r = (position_i - position_j); 
    r.w = 0.f; // I stored density in 4th component
    // |r|
    float rlen = length(r);
    //pt->density.y += 1.;

    // is this particle within cutoff?
    if (rlen <= particle_sphp->smoothing_distance)
    {
        // return density.x for single neighbor
        //float Wij = sphp->wpoly6_coef * Wpoly6(r, sphp->smoothing_distance, sphp);
        float Wij = Wpoly6(r, particle_sphp->smoothing_distance, sphp);

        //float casper = ghost_intensity[index_j].w;
        //float ghost_factor = casper(ghost_intensity[index_j].w, 1.1);
        //float ghost_factor = casper_square(ghost_intensity[index_j].w, .5);
        //float ghost_factor = casper_cubic(ghost_intensity[index_j].w, target_intensity);
        float ghost_factor = casper_poly6(ghost_intensity[index_j].w, target_intensity);
        pt->density.x += sphp->mass*Wij * ghost_factor;
        //pt->density.x += sphp->mass*Wij * (1.5 - casper)*(1.5-casper)*(1.5-casper);
        //pt->density.x += sphp->mass*Wij;
    }
}
//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"
//--------------------------------------------------------------
// compute forces on particles

__kernel void ghost_density_update(
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
    int num = particle_sphp->num;
    //int numParticles = get_global_size(0);
    //int num = get_global_size(0);

    int index = get_global_id(0);
    if (index >= num) return;

#if 1
    float4 position_i = pos[index] * sphp->simulation_scale;

    //debuging
    clf[index] = (float4)(99,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

    //IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    IterateParticlesInNearbyCells(ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    
    //doing this makes particles in non-target region go crazy (when doing ghost forces only using ghost_density
    //density[index] -= sphp->wpoly6_coef * pt.density.x* .000001f;


    ghost_density[index] = sphp->wpoly6_coef * pt.density.x * .0001f;
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

