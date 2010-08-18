/**
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 */

#ifndef _SRTS_RADIX_SORT_DRIVER_H_
#define _SRTS_RADIX_SORT_DRIVER_H_



#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <cutil.h>


//------------------------------------------------------------------------------
// Sorting includes
//------------------------------------------------------------------------------

// Kernel includes
#include <srts_radix_sort_kernel.cu>


//------------------------------------------------------------------------------
// Defines, constants, globals 
//------------------------------------------------------------------------------

const unsigned int DEFAULT_GRID_SIZE		= 150;		// default max grid size (for GT200)


//------------------------------------------------------------------------------
// Routines
//------------------------------------------------------------------------------


/**
 * Heuristic for determining the number of CTAs to launch.
 *   
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		max_grid_size  
 * 		Maximum allowable number of CTAs to launch.  A value of -1 indicates 
 * 		that the default value should be used.
 * 
 * @return The actual number of CTAs that should be launched
 */
unsigned int  SelectGridSize(
	unsigned int  num_elements, 
	bool verbose) 
{

	unsigned int max_grid_size = DEFAULT_GRID_SIZE;
	
	if ((num_elements + SRTS_CYCLE_ELEMENTS - 1) / SRTS_CYCLE_ELEMENTS >= DEFAULT_GRID_SIZE) {

		uint attempts = 0;
		uint multiplier = 16;

		double top_delta = 0.078;	
		double bottom_delta = 0.078;

		uint dividend = (num_elements + SRTS_CYCLE_ELEMENTS - 1) / SRTS_CYCLE_ELEMENTS;

		while(true) {

			double quotient = ((double) dividend) / (multiplier * max_grid_size);
			quotient -= (int) quotient;

			if (verbose) printf("%d, %d, %d, %f @ %f/%f deltas\n", max_grid_size, dividend, multiplier, quotient, top_delta, bottom_delta);

			if ((quotient > top_delta) && (quotient < 1 - bottom_delta)) {
				break;
			}

			if (max_grid_size == 147) {
				max_grid_size = 120;
			} else {
				max_grid_size -= 1;
			}

			attempts++;
		}
	}

	// Calculate the actual number of threadblocks to launch.  Initially
	// assume that each threadblock will do only one cycle_elements worth 
	// of work, but then clamp it by the "max" restriction derived above
	// in order to accomodate the "single-sp" and "saturated" cases.

	uint grid_size = (num_elements + SRTS_CYCLE_ELEMENTS - 1) / SRTS_CYCLE_ELEMENTS;
	if (grid_size == 0) {
		grid_size = 1;
	}
	if (grid_size > max_grid_size) {
		grid_size = max_grid_size;
	} 

	return grid_size;
}



template <typename K, typename V, unsigned int RADIX_BITS, unsigned int BIT>
void SortDigit(
	bool verbose,
	unsigned int num_elements,
	unsigned int grid_size,
	unsigned int shared_mem_size,
	K* d_in_keys,
	K* d_out_keys,
	V* d_in_data,
	V* d_out_data,
	unsigned int *d_spine,
	unsigned int num_big_blocks,
	unsigned int big_block_elements,
	unsigned int normal_block_elements,
	unsigned int extra_elements_last_block,
	unsigned int spine_block_elements)
{
	// Check for any launch errors
	CUT_CHECK_ERROR("Kernel execution failed.");

	//
	// Bottom-level reduction kernel
	//
	
	// Run flush kernel if we have two or more threadblocks for each of the SMs
	if (num_elements > 60 * 512) FlushKernel<<<grid_size, SRTS_THREADS, 3000>>>();
	
	if (verbose) {
		printf("TreeReduce <<<%d,%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d)\n\n",
			grid_size, SRTS_THREADS, shared_mem_size,
			SRTS_CYCLE_ELEMENTS,
			num_big_blocks,
			big_block_elements,
			normal_block_elements,
			extra_elements_last_block);
	}

	TreeReduce<K, RADIX_BITS, BIT> 
		<<<grid_size, SRTS_THREADS, shared_mem_size>>>(
			d_in_keys,
			d_spine,
			num_big_blocks,
			big_block_elements,
			normal_block_elements,
			extra_elements_last_block);

	// Check for any launch errors
	CUT_CHECK_ERROR("Kernel execution failed.");

	//
	// Top-level scan kernel
	//
	
	if (verbose) {
		printf("SrtsScanSpine<<<%d,%d,%d>>>(\n\tspine_block_elements: %d)\n\n", 
			1, SRTS_THREADS, shared_mem_size, 
			spine_block_elements);
	}

	SrtsScanSpine<<<1, SRTS_THREADS, shared_mem_size>>>(
		d_spine,
		d_spine,
		spine_block_elements);

	// Check for any launch errors
	CUT_CHECK_ERROR("Kernel execution failed.");
	
	//
	// Bottom-level scan/distribute kernel
	//
	
	// Run flush kernel if we have two or more threadblocks for each of the SMs
	if (num_elements > 60 * 512) FlushKernel<<<grid_size, 128, 3000>>>();

	if (verbose) {
		printf("SrtsScanDigitBulk <<<%d,%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d,\n\textra_elements_last_block: %d)\n\n", 
			grid_size, SRTS_THREADS, shared_mem_size, 
			SRTS_CYCLE_ELEMENTS,
			num_big_blocks,
			big_block_elements,
			normal_block_elements, 
			extra_elements_last_block);
	}

	if (d_in_data == NULL) {
	
		// keys only 
		SrtsScanDigitBulk<K, V, true, RADIX_BITS, BIT> 					
		<<<grid_size, SRTS_THREADS, shared_mem_size>>>(
			d_spine,
			d_in_keys,
			d_out_keys,
			d_in_data,
			d_out_data,
			num_big_blocks,
			big_block_elements,
			normal_block_elements,
			extra_elements_last_block);
		
	} else {

		// keys and values
		SrtsScanDigitBulk<K, V, false, RADIX_BITS, BIT> 					
		<<<grid_size, SRTS_THREADS, shared_mem_size>>>(
			d_spine,
			d_in_keys,
			d_out_keys,
			d_in_data,
			d_out_data,
			num_big_blocks,
			big_block_elements,
			normal_block_elements,
			extra_elements_last_block);
	}
		
	// Check for any launch errors
	CUT_CHECK_ERROR("Kernel execution failed.");
}


/**
 * Launches a simple, two-level sort.  
 * 
 * template-param K
 * 		Type of keys to be sorted
 * template-param V
 * 		Type of values to be sorted
 *
 * @param[in] 	num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 	d_in_keys 
 * 		Input device vector of keys to sort
 * @param[in] 	d_out_keys 
 * 		Input device vector of sorted keys.
 * @param[in] 	d_in_data  
 * 		Input device vector of values to sort.  May be null.
 * @param[in] 	d_out_data  
 * 		Output device vector of sorted elements.   May be null.
 * @param[in/out]	d_spine  
 * 		Pointer to temporary device storage needed for radix sort.  (Is used to 
 *  	orchestrate coordination between bottom-level CTAs.) If NULL, one will be 
 *      allocated by this routine (and must be subsequently cuda-freed by the caller)
 * @param[in] 	verbose  
 * 		Flag whether or not to print launch information to stdout
 * 
 * @return true if results are in d_out_keys, false if they are in d_in_keys
 */
template <typename K, typename V>
bool LaunchSort(
	unsigned int num_elements, 
	K* d_in_keys, 
	K* d_out_keys, 
	V* d_in_data, 
	V* d_out_data, 
	unsigned int **p_d_spine, 
	bool verbose) 
{
	//
	// Determine number of CTAs to launch, shared memory, cycle elements, etc.
	//
	
	unsigned int grid_size = 			SelectGridSize(num_elements, verbose);
	unsigned int shared_mem_size = 		0;
	unsigned int cycle_elements = 		SRTS_CYCLE_ELEMENTS;
	

	//
	// Determine how many elements each CTA will process
	//
	// A given threadblock may receive one of three different amounts of 
	// work: "big", "normal", and "last".  The big workloads are one
	// cycle_elements greater than the normal, and the last workload 
	// does the extra (problem-size % cycle_elements) work.
	//

	unsigned int total_cycles = 
		(num_elements + cycle_elements - 1) / cycle_elements;
	unsigned int cycles_per_block =
		total_cycles / grid_size;						
	unsigned int extra_cycles = 
		total_cycles - (cycles_per_block * grid_size);
	unsigned int extra_elements_last_block = 
		num_elements - ((num_elements / cycle_elements) * cycle_elements);

	unsigned int normal_block_elements = cycles_per_block * cycle_elements;
	unsigned int num_big_blocks = extra_cycles;
	unsigned int big_block_elements = (cycles_per_block + 1) * cycle_elements;

	
	//
	// Determine number of elements (and cycles) for the top-level spine scan (round up)
	//
	
	unsigned int spine_cycles = ((grid_size * (1 << 4)) + SRTS_CYCLE_ELEMENTS - 1) / 		// 4 radix bits here
			SRTS_CYCLE_ELEMENTS;

	unsigned int spine_block_elements = 
		spine_cycles * SRTS_CYCLE_ELEMENTS;
	

	//
	// Allocate and initialize device memory for spine
	//

	if (*p_d_spine == NULL) {
		unsigned int spine_size = sizeof(unsigned int) * spine_block_elements;
		CUDA_SAFE_CALL( cudaMalloc((void**) p_d_spine, spine_size) );
	}
	
	
	//
	// Sort using 4-bit radix digit passes  
	//
	
	SortDigit<K, V, 4, 0> (verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
	verbose = false;
	SortDigit<K, V, 4, 4> (verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 

	if (sizeof(K) > 1) {
		SortDigit<K, V, 4, 8>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 12>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
	}
	if (sizeof(K) > 2) {
		SortDigit<K, V, 4, 16>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 20>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
	}
	if (sizeof(K) > 3) {
		SortDigit<K, V, 4, 24>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 28>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements);
	}
	if (sizeof(K) > 4) {
		SortDigit<K, V, 4, 32>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 36>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements);
	}
	if (sizeof(K) > 5) {
		SortDigit<K, V, 4, 40>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 44>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements);
	}
	if (sizeof(K) > 6) {
		SortDigit<K, V, 4, 48>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 52>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements);
	}
	if (sizeof(K) > 7) {
		SortDigit<K, V, 4, 56>(verbose, num_elements, grid_size, shared_mem_size, d_in_keys, d_out_keys, d_in_data, d_out_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements); 
		SortDigit<K, V, 4, 60>(verbose, num_elements, grid_size, shared_mem_size, d_out_keys, d_in_keys, d_out_data, d_in_data, *p_d_spine, num_big_blocks, big_block_elements, normal_block_elements, extra_elements_last_block, spine_block_elements);
	}
	
	//
	// The output is back in d_in_keys. (It seems weird, but consider if you did 3-bit passes
	// instead: for 32-bit keys, you'd have ten 3-bit passes and one 2-bit pass, which
	// leaves the results in the d_out_keys vector.  In that case, we'd return true.)
	//
	
	return false;
}


/**
 * Launches a simple, **keys-only** two-level sort.  
 * 
 * template-param K
 * 		Type of keys to be sorted
 *
 * @param[in] 	num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 	d_in_keys 
 * 		Input device vector of keys to sort
 * @param[in] 	d_out_keys 
 * 		Input device vector of sorted keys.
 * @param[in/out]	d_spine  
 * 		Pointer to temporary device storage needed for radix sort.  (Is used to 
 *  	orchestrate coordination between bottom-level CTAs.) If NULL, one will be 
 *      allocated by this routine (and must be subsequently cuda-freed by the caller)
 * @param[in] 	verbose  
 * 		Flag whether or not to print launch information to stdout
 * 
 * @return true if results are in d_out_keys, false if they are in d_in_keys
 */
template <typename K>
bool LaunchSort(
	unsigned int num_elements, 
	K* d_in_keys, 
	K* d_out_keys, 
	unsigned int **p_d_spine, 
	bool verbose) 
{
	
	return LaunchSort<K, unsigned int>(
		num_elements, 
		d_in_keys, 
		d_out_keys,
		NULL, 
		NULL, 
		p_d_spine, 
		verbose);
}


#endif

