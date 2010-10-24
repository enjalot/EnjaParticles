// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"

//----------------------------------------------------------------------
__kernel void datastructures(
					__global float4*   vars_unsorted,
					__global float4*   vars_sorted, 
		   			__global int* sort_hashes,
		   			__global int* sort_indices,
		   			__global int* cell_indices_start,
		   			__global int* cell_indices_end,
		   			__global int* cell_indices_nb, // NOT USED
		   			__constant struct SPHParams* sphp,
		   			__constant struct GridParams* gp,
					__local  int* sharedHash   // blockSize+1 elements
			  )
{
	int index = get_global_id(0);
	int numParticles = get_global_size(0);


	// particle index	
	if (index >= numParticles) return;

	int hash = sort_hashes[index];

	// Load hash data into shared memory so that we can look 
	// at neighboring particle's hash value without loading
	// two hash values per thread	

	int tid = get_local_id(0);

#if 1
	sharedHash[tid+1] = hash;  // SOMETHING WRONG WITH hash on Fermi

	if (index > 0 && tid == 0) {
		// first thread in block must load neighbor particle hash
		sharedHash[0] = sort_hashes[index-1];
	}

#ifndef __DEVICE_EMULATION__
	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	// If this particle has a different cell index to the previous
	// particle then it must be the first particle in the cell,
	// so store the index of this particle in the cell.
	// As it isn't the first particle, it must also be the cell end of
	// the previous particle's cell

	if ((index == 0 || hash != sharedHash[tid]) )
	{
		cell_indices_start[hash] = index; // ERROR
		if (index > 0) {
			cell_indices_end[sharedHash[tid]] = index;
		}
	}
	//return;

	if (index == numParticles - 1) {
		cell_indices_end[hash] = index + 1;
	}

	int sorted_index = sort_indices[index];

	// Copy data from old unsorted buffer to sorted buffer

	#if 0
	int nb_vars = gp->nb_vars;
	for (int j=0; j < nb_vars; j++) {
		vars_sorted[index+j*numParticles]	= vars_unsorted[sorted_index+j*numParticles];
	}
	#endif


	// Variables to sort could change for different types of simulations 
	// SHOULD I divide by simulation scale upon return? do not think so
	pos(index)     = unsorted_pos(sorted_index) * sphp->simulation_scale;
	vel(index)     = unsorted_vel(sorted_index);
	veleval(index) = unsorted_veleval(sorted_index); // not sure if needed
	//density(index) = unsorted_density(sorted_index); // only for debugging
#endif
}
//----------------------------------------------------------------------

#endif
