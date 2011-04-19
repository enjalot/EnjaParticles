#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float4* ghost_pos, __global float* density, __global float* ghost_density, __global float4* ghost_intensity, __global float4* veleval, __global float4* force, __global float4* xsph, __constant struct SPHParams* particle_sphp
#define ARGV pos, ghost_pos, density, ghost_density, ghost_intensity, veleval, force, xsph, particle_sphp

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

    // is this particle within cutoff?
    if (rlen <= particle_sphp->smoothing_distance)
    {

        // update pressure
        // gradient
        int iej = 1;
        int rlencheck = rlen != 0.;
        iej *= rlencheck;

        float casper = ghost_intensity[index_j].x;

        float dWijdr = Wspiky_dr(rlen, particle_sphp->smoothing_distance, sphp);

        float di = ghost_density[index_i];  // should not repeat di=
        float dj = 1000 * (1.7 - casper);
        float idi = 1.0/di;
        float idj = 1.0/dj;

        //form simple SPH in Krog's thesis

        float rest_density = 1000.f;
        float Pi = sphp->K*(di - rest_density);
        float Pj = sphp->K*(dj - rest_density);

        
        float kern = -.5 * dWijdr * (Pi + Pj) * sphp->wspiky_d_coef;
        //float kern = dWijdr * (Pi * idi * idi + Pj * idj * idj) * sphp->wspiky_d_coef;
        float4 force = kern*r; 

        //float4 force = (float4)(di, dj, di*dj, 0,);


        float4 veli = veleval[index_i]; // sorted
        float4 velj = -veli;
         
        
#if 0
       // Add viscous forces
        float vvisc = sphp->viscosity;
        float dWijlapl = sphp->wvisc_dd_coef * Wvisc_lapl(rlen, particle_sphp->smoothing_distance, sphp);
        force += vvisc * (velj-veli) * dWijlapl;
#endif

        //force *= sphp->mass/(di*dj);  // original
        force *= sphp->mass/(di*dj) * (1.5f - casper);
        ///force *= sphp->mass/(di*dj); 

#if 1
        // Add XSPH stabilization term
        // the poly6 kernel calculation seems to be wrong, using rlen as a vector when it is a float...
        //float Wijpol6 = Wpoly6(r, particle_sphp->smoothing_distance, sphp) * sphp->wpoly6_coeff;
        /*
        float h = sphp->smoothing_distance;
        float hr2 = (h*h - rlen*rlen);
        float Wijpol6 = hr2*hr2*hr2;// * sphp->wpoly6_coeff;
        */
        float Wijpol6 = Wpoly6(r, particle_sphp->smoothing_distance, sphp);
        //float Wijpol6 = sphp->wpoly6_coef * Wpoly6(rlen, sphp->smoothing_distance, sphp);
        float4 xsph = (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di+dj)) * (1.5f-casper);
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

__kernel void ghost_force_update(
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
    int num = particle_sphp->num;
    //int numParticles = get_global_size(0);
    //int num = get_global_size(0);

    int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos[index] * sphp->simulation_scale;

    //debuging
    clf[index] = (float4)(0,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

    //IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    IterateParticlesInNearbyCells(ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp DEBUG_ARGV);
    force[index] += pt.force; 
    clf[index].xyz = pt.force.xyz;
    xsph[index] += sphp->wpoly6_coef * pt.xsph * .00001f;
}

/*-------------------------------------------------------------- */
#endif

