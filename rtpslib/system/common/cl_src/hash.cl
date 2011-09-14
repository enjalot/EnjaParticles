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


#ifndef _HASH_CL_H_
#define _HASH_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"
#include "cl_hash.h"

//----------------------------------------------------------------------
// Calculate a grid hash value for each particle


//  Have to make sure that the data associated with the pointers is on the GPU
//struct GridData
//{
//    uint* sort_hashes;          // particle hashes
//    uint* sort_indexes;         // particle indices
//    uint* cell_indexes_start;   // mapping between bucket hash and start index in sorted list
//    uint* cell_indexes_end;     // mapping between bucket hash and end index in sorted list
//};

//----------------------------------------------------------------------
// comes from K_Grid_Hash
// CANNOT USE references to structures/classes as arguments!
__kernel void hash(
                  //__global float4* vars_unsorted,
                  int num,
                  __global float4* pos_u,
                  __global uint* sort_hashes,
                  __global uint* sort_indexes,
                  //__constant struct SPHParams* sphp,
                  __constant struct GridParams* gp
                  DEBUG_ARGS
                  //__global float4* fdebug,
                  //__global int4* idebug
                  )
{
    // particle index
    uint index = get_global_id(0);
    //int num = sphp->num;
    //int num = get_global_size(0);
    //comment this out to hash everything if using max_num
    if (index >= num) return;

    // initialize to -1 (used in kernel datastructures in build_datastructures_wrap.cpp
    //int grid_size = (int) (gp->grid_res.x*gp->grid_res.y*gp->grid_res.z);
    //if (index < grid_size) {   // grid_size: 1400
    //cell_indices_start[index] = 0xffffffff; 
    //}

    // particle position
    //float4 p = unsorted_pos(index); // macro
    float4 p = pos_u[index]; // macro

    // get address in grid
    //int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_inv_delta);
    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_delta);
    bool wrap_edges = false;
    //uint hash = (uint) calcGridHash(gridPos, gp->grid_res, wrap_edges);//, fdebug, idebug);
    int hash = calcGridHash(gridPos, gp->grid_res, wrap_edges);//, fdebug, idebug);

    cli[index].xyz = gridPos.xyz;
    cli[index].w = hash;
    //cli[index].w = (gridPos.z*gp->grid_res.y + gridPos.y) * gp->grid_res.x + gridPos.x; 

    hash = hash > gp->nb_cells ? gp->nb_cells : hash;
    hash = hash < 0 ? gp->nb_cells : hash;
    /*
       //problem is that when we cut num we are hashing the wrong stuff?
    if (index >= num)
    {
        hash = gp->nb_cells;
    }
    */
    // store grid hash and particle index
    sort_hashes[index] = (uint)hash;
    //int pp = (int) p.x;

    sort_indexes[index] = index;

    //fdebug[index] = gp->grid_inv_delta;
    //fdebug[index] = (float4)((p.x - gp->grid_min.x) * gp->grid_inv_delta.x, p.x, 0,0);
    //clf[index] = (float4)((p.x - gp->grid_min.x) * gp->grid_delta.x, p.x, 0,0);
    //clf[index] = p;
    //cli[index].w = sphp->max_num;


/*
    clf[0].x = sphp->mass;
    clf[0].y = sphp->rest_distance;
    clf[0].z = sphp->smoothing_distance;
    clf[0].w = sphp->simulation_scale;

    clf[1].x = sphp->boundary_stiffness;
    clf[1].y = sphp->boundary_dampening;
    clf[1].z = sphp->boundary_distance;
    clf[1].w = sphp->EPSILON;

    clf[2].x = sphp->PI;       //delicious
    clf[2].y = sphp->K;        //speed of sound
    clf[2].z = sphp->viscosity;
    clf[2].w = sphp->velocity_limit;

    clf[3].x = sphp->xsph_factor;
    clf[3].y = sphp->gravity; // -9.8 m/sec^2
    clf[3].z = sphp->friction_coef;
    clf[3].w = sphp->restitution_coef;

    clf[4].x = sphp->shear;
    clf[4].y = sphp->attraction;
    clf[4].z = sphp->spring;
    //kernel coefficients
    clf[4].w = sphp->wpoly6_coef;

    clf[5].x = sphp->wpoly6_d_coef;
    clf[5].y = sphp->wpoly6_dd_coef; // laplacian
    clf[5].z = sphp->wspiky_coef;
    clf[5].w = sphp->wspiky_d_coef;

    clf[6].x = sphp->wspiky_dd_coef;
    clf[6].y = sphp->wvisc_coef;
    clf[6].z = sphp->wvisc_d_coef;
    clf[6].w = sphp->wvisc_dd_coef;

    clf[7].x = sphp->num;
    clf[7].y = sphp->nb_vars; // for combined variables (vars_sorted, etc.)
    clf[7].z = sphp->choice; // which kind of calculation to invoke
    clf[7].w = sphp->max_num;
*/



}
//----------------------------------------------------------------------


#endif
