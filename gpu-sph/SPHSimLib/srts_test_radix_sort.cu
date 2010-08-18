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


#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <cutil.h>


//------------------------------------------------------------------------------
// Sorting includes
//------------------------------------------------------------------------------

// Sorting includes
#include <srts_radix_sort.cu>

// For correctness-checking only
#include <srts_verifier.cu>


//------------------------------------------------------------------------------
// Defines, constants, globals 
//------------------------------------------------------------------------------

unsigned int g_timer;

bool g_verbose;
bool g_verbose2;
bool g_verify;
bool g_measure_cpu;
bool g_keys_only;


//------------------------------------------------------------------------------
// Routines
//------------------------------------------------------------------------------


/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\nsrts_radix_sort [--noprompt] [--v[2]] [--i=<num-iterations>] [--n=<num-elements>] [--keys-only]\n"); 
	printf("\n");
	printf("\t--v\tDisplays kernel launch config info.\n");
	printf("\n");
	printf("\t--v2\tSame as --v, but displays the results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-iterations> times\n");
	printf("\t\t\ton the device.  (Only copies input/output once.) Default=1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample \n");
	printf("\t\t\tproblem.  Default=512\n");
	printf("\t--keys-only\tSpecifies that keys are not accommodated by value pairings\n");
	printf("\t\t\tproblem.  Default=512\n");
	printf("\n");
}


/**
 * Uses the GPU to sort the specified vector of elements for the given 
 * number of iterations, displaying runtime information.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in,out] 	h_data  
 * 		Vector of values to sort (may be null)
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K, typename V>
void SortFromHost(
	unsigned int num_elements, 
	K *h_keys,
	V *h_data, 
	unsigned int iterations
) 
{
	// Place holder for temporary spine data (will get allocated upon first use)
	unsigned int *d_spine = NULL;
	bool results_placed_in_output;
	
	//
	// Allocate and initialize device memory for keys
	//

	K* d_in_keys;
	K* d_out_keys;
    unsigned int keys_mem_size = sizeof(K) * num_elements;
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_in_keys, keys_mem_size) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_out_keys, keys_mem_size) );
	CUDA_SAFE_CALL( cudaMemcpy(d_in_keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );

	// Initialize output device memory (not necessary, but can help with debugging
	// Todo: remove
	for (int i = 0; i < num_elements; i++) {
		h_keys[i] = 9999;
	}
	CUDA_SAFE_CALL( cudaMemcpy(d_out_keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );

	//
	// Allocate and initialize device memory for data
	//

	V* d_in_data = NULL;
	V* d_out_data = NULL;
	unsigned int data_mem_size = sizeof(V) * num_elements;
	if (h_data != NULL) {
		CUDA_SAFE_CALL( cudaMalloc((void**) &d_in_data, data_mem_size) );
		CUDA_SAFE_CALL( cudaMalloc((void**) &d_out_data, data_mem_size) );
		CUDA_SAFE_CALL( cudaMemcpy(d_in_data, h_data, data_mem_size, cudaMemcpyHostToDevice) );
	}

	//
	// Perform the timed number of sorting iterations
	//
	
	// Make sure there are no CUDA errors before we launch
	CUT_CHECK_ERROR("Kernel execution failed (errors before launch)");

	// Start the timer and perform the scan iterations
	CUT_SAFE_CALL( cutCreateTimer(&g_timer) );
	CUT_SAFE_CALL( cutStartTimer(g_timer) );
	for (int i = 0; i < iterations; i++) {

		if (g_keys_only) {
		
			results_placed_in_output = LaunchSort<K>(
				num_elements, 
				d_in_keys, 
				d_out_keys,
				&d_spine, 
				(g_verbose && (i == 0)));

		} else {

			results_placed_in_output = LaunchSort<K, V>(
				num_elements, 
				d_in_keys, 
				d_out_keys,
				d_in_data, 
				d_out_data, 
				&d_spine, 
				(g_verbose && (i == 0)));
		}
	}
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer(g_timer) );


	//
	// Display timing information
	//
	
	double avg_runtime = cutGetTimerValue(g_timer) / iterations;
	double throughput = ((double) num_elements) / 1000.0 / 1000.0 / cutGetTimerValue(g_timer) * ((double) iterations); 
    CUT_SAFE_CALL( cutDeleteTimer(g_timer) );

    printf("%d iterations, %d elements, %f GPU ms, %f x10^9 elts/sec\n", 
		iterations, 
		num_elements,
		avg_runtime,
		throughput);
    
	
    // 
    // Copy out data & free allocated memory
    //
    
	// Sorted keys 
	if (results_placed_in_output) {
		CUDA_SAFE_CALL( cudaMemcpy(h_keys, d_out_keys, keys_mem_size, cudaMemcpyDeviceToHost) );
	} else {
		CUDA_SAFE_CALL( cudaMemcpy(h_keys, d_in_keys, keys_mem_size, cudaMemcpyDeviceToHost) );
	}
    CUDA_SAFE_CALL( cudaFree(d_in_keys) );
    CUDA_SAFE_CALL( cudaFree(d_out_keys) );

	if (h_data != NULL) {
		// Sorted values back to host
		if (results_placed_in_output) {
			CUDA_SAFE_CALL( cudaMemcpy(h_data, d_out_data, data_mem_size, cudaMemcpyDeviceToHost) );
		} else {
			CUDA_SAFE_CALL( cudaMemcpy(h_data, d_in_data, data_mem_size, cudaMemcpyDeviceToHost) );
		}
	    CUDA_SAFE_CALL( cudaFree(d_in_data) );
	    CUDA_SAFE_CALL( cudaFree(d_out_data) );
	}

	// free spine
    CUDA_SAFE_CALL( cudaFree(d_spine) );

}


/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of K elements, values of V elements, and then dispatches the problem 
 * to the GPU for the given number of iterations, displaying runtime information.
 *
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 */
template<typename K, typename V>
void TestSort(
	unsigned int iterations,
	int num_elements)
{
	K *h_keys;
	V *h_data;

	//
    // Allocate the sorting problem on the host
	//

	h_keys = (K*) malloc(num_elements * sizeof(K));
	h_data = (V*) malloc(num_elements * sizeof(V));

	
	//
	// Fill the keys with random bytes
	//
	
	unsigned char key_bytes[sizeof(K)];
//	srand(time(NULL));
	srand(0);
	for (unsigned int i = 0; i < num_elements; ++i) {
    	for (unsigned int j = 0; j < sizeof(K); j++) {
    		key_bytes[j] = (unsigned char) (rand() >> 8);
    	}
    	
    	memcpy(&h_keys[i], key_bytes, sizeof(K));
	}
    
    // 
    // Run the primitive 
    //
    
    SortFromHost<K, V>(
    	num_elements, 
    	h_keys,
    	h_data, 
    	iterations);
    
	
    //
    // Verify solution
    //
    
    if (g_verify) VerifySort<K>(h_keys, num_elements, g_verbose);
	printf("\n");
	fflush(stdout);
	
	
	//
	// Display sorted key data
	//
	
	if (g_verbose2) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<K>(h_keys[i]);
			printf(", ");
		}
		printf("\n");
	}	
	
	
	//
	// Free our allocated host memory 
	//
	
    free(h_keys);
    free(h_data);
}


//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main( int argc, char** argv) {

	CUT_DEVICE_INIT(argc, argv);

    unsigned int num_elements 					= 512;
    unsigned int iterations  					= 1;

    //
	// Check command line arguments
    //

    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}

    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&iterations);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
	g_keys_only = cutCheckCmdLineFlag( argc, (const char**) argv, "keys-only");
	if (g_verbose2 = cutCheckCmdLineFlag( argc, (const char**) argv, "v2")) {
		g_verbose = true;
	} else {
		g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");
	}
	g_verify = !cutCheckCmdLineFlag( argc, (const char**) argv, "noverify");
	
	// Execute test
	TestSort<unsigned long, unsigned long>(
		iterations,
		num_elements);

	CUT_EXIT(argc, argv);
}



