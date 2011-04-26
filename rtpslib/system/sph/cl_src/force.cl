#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float* density, __global float4* veleval, __global float4* force, __global float4* xsph
#define ARGV pos, density, veleval, force, xsph

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
    float4 r = (position_i - position_j); 
    r.w = 0.f; // I stored density in 4th component
    // |r|
    float rlen = length(r);

    // is this particle within cutoff?
    if (rlen <= sphp->smoothing_distance)
    {

        //iej is 0 when we are looking at same particle
        //we allow calculations and just multiply force and xsph
        //by iej to avoid branching
        int iej = index_i != index_j;

        // avoid divide by 0 in Wspiky_dr
        rlen = max(rlen, sphp->EPSILON);

        float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

        float di = density[index_i];  // should not repeat di
        float dj = density[index_j];
        float idi = 1.0/di;
        float idj = 1.0/dj;

        //form simple SPH in Krog's thesis

        float rest_density = 1000.f;
        float Pi = sphp->K*(di - rest_density);
        float Pj = sphp->K*(dj - rest_density);

        float kern = -.5 * dWijdr * (Pi + Pj) * sphp->wspiky_d_coef * idi * idj;
        //float kern = -1.0f * dWijdr * (Pi * idi * idi + Pj * idj * idj) * sphp->wspiky_d_coef;
        float4 force = kern*r; 

        float4 veli = veleval[index_i]; // sorted
        float4 velj = veleval[index_j];

#if 1
        // Add viscous forces
        float vvisc = sphp->viscosity;
        float dWijlapl = sphp->wvisc_dd_coef * Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
        force += vvisc * (velj-veli) * dWijlapl * idj;
#endif

        //force *=  sphp->mass/(di.x*dj.x);  // original
        force *=  sphp->mass;// * idi * idj;
        //force *=  sphp->mass;// /(di.x*dj.x); 

#if 1
        // Add XSPH stabilization term
        // the poly6 kernel calculation seems to be wrong, using rlen as a vector when it is a float...
        //float Wijpol6 = Wpoly6(r, sphp->smoothing_distance, sphp) * sphp->wpoly6_coeff;
        /*
        float h = sphp->smoothing_distance;
        float hr2 = (h*h - rlen*rlen);
        float Wijpol6 = hr2*hr2*hr2;// * sphp->wpoly6_coeff;
        */
        float Wijpol6 = Wpoly6(r, sphp->smoothing_distance, sphp);
        //float Wijpol6 = sphp->wpoly6_coef * Wpoly6(rlen, sphp->smoothing_distance, sphp);
        float4 xsph = (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di+dj));
        pt->xsph += xsph * (float)iej;
        pt->xsph.w = 0.f;
#endif

        pt->force += force * (float)iej;

    }
}

//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"

//--------------------------------------------------------------
// compute forces on particles

__kernel void force_update(
                       //__global float4* vars_sorted,
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

    float4 position_i = pos[index] * sphp->simulation_scale;

    //debuging
    clf[index] = (float4)(99,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

    //IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    IterateParticlesInNearbyCells(ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    force[index] = pt.force; 
    clf[index].xyz = pt.force.xyz;
    xsph[index] = sphp->wpoly6_coef * pt.xsph;
}

/*-------------------------------------------------------------- */
#endif

