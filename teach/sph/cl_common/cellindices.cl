// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _CELLINDICES_
#define _CELLINDICES_ 

#include "cl_macros.h"
#include "cl_structs.h"

//#pragma cl_khr_global_int32_base_atomics : enable
//----------------------------------------------------------------------
__kernel void cellindices(
                            int num,
                            __global uint* sort_hashes,
                            __global uint* sort_indices,
                            __global uint* cell_indices_start,
                            __global uint* cell_indices_end,
                            //__constant struct SPHParams* sphp,
                            __constant struct GridParams* gp,
                            __local  uint* sharedHash   // blockSize+1 elements
                            )
{
    uint index = get_global_id(0);
    //int num = sphp->num;
    //int num = get_global_size(0);
    //if (index >= num) return;
    uint ncells = gp->nb_cells;

    uint hash = sort_hashes[index];

    //don't want to write to cell_indices arrays if hash is out of bounds
    if( hash > ncells)
    {
        return;
    }
#if 1
    // Load hash data into shared memory so that we can look 
    // at neighboring particle's hash value without loading
    // two hash values per thread	

    uint tid = get_local_id(0);
    //if(tid >= 64) return;

    sharedHash[tid+1] = hash;  // SOMETHING WRONG WITH hash on Fermi

    if (index > 0 && tid == 0)
    {
        // first thread in block must load neighbor particle hash
        uint hashm1 = sort_hashes[index-1] < ncells ? sort_hashes[index-1] : ncells;
        sharedHash[0] = hashm1;
        //sharedHash[0] = sort_hashes[index-1];
        /*
        if(hash >= gp->nb_cells-1) //if particles go out of bounds, delete them
        {
            //cell_indices_end[gp->nb_cells - 2] = index + 1; //make sure last cell index is right // this is totally confused
            if(num_changed[0] == 0)
            {
                num_changed[0] = index; //new number of particles to use
                //num = index;
                return;
            }
        }
        */

    }

#ifndef __DEVICE_EMULATION__
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    //Having this check here is important! Can't quit before local threads are done
    //but we can't keep going if our index goes out of bounds of the number of particles
    if (index >= num) return;

    //if(hash < gp->nb_cells)
    //{
    //if ((index == 0 || hash != sharedHash[tid]))
    if (index == 0)
    {
        cell_indices_start[hash] = index;
    }

    if (index > 0)
    {
        if(sharedHash[tid] != hash)
        {
            cell_indices_start[hash] = index; 
            cell_indices_end[sharedHash[tid]] = index;
        }
    }
    //return;

    if (index == num - 1)
    {
        cell_indices_end[hash] = index + 1;
    }
    //}
    

#endif
}
//----------------------------------------------------------------------

#endif
