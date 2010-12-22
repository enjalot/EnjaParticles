// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _GHOSTDATASTRUCTURES_
#define _GHOSTDATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structs.h"

//----------------------------------------------------------------------
__kernel void ghost_datastructures(
                    uint num,
                    __global float4*   ghosts,
                    __global float4*   ghosts_sorted, 
                    __global uint* sort_hashes,
                    __global uint* sort_indices,
                    __constant struct SPHParams* sphp,
                    //__constant struct GridParams* gp,
                    __local  uint* sharedHash   // blockSize+1 elements
              )
{
    uint index = get_global_id(0);
    //int num = get_global_size(0);


    // particle index	
    if (index >= num) return;

    uint hash = sort_hashes[index];

    // Load hash data into shared memory so that we can look 
    // at neighboring particle's hash value without loading
    // two hash values per thread	

    uint tid = get_local_id(0);
    //if(tid >= 64) return;

#if 1
    sharedHash[tid+1] = hash;  // SOMETHING WRONG WITH hash on Fermi

    if (index > 0 && tid == 0) {
        // first thread in block must load neighbor particle hash
        sharedHash[0] = sort_hashes[index-1];
    }

#ifndef __DEVICE_EMULATION__
    barrier(CLK_LOCAL_MEM_FENCE);
#endif


    uint sorted_index = sort_indices[index];
    //uint sorted_index = index;

    // Copy data from old unsorted buffer to sorted buffer

    #if 0
    int nb_vars = gp->nb_vars;
    for (int j=0; j < nb_vars; j++) {
        vars_sorted[index+j*numParticles]	= vars_unsorted[sorted_index+j*numParticles];
    }
    #endif

    // Variables to sort could change for different types of simulations 
    ghosts_sorted[index]     = ghosts[sorted_index] * sphp->simulation_scale;
#endif
}
//----------------------------------------------------------------------

#endif
