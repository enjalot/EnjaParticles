#include "SPH.h"

#include <string.h>
/*
#include "oclSortingNetworks_common.h"


extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);

extern "C" void closeBitonicSort(void);

extern"C" size_t bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal,
    uint batch,
    uint arrayLength,
    uint dir
);
*/



namespace rtps
{

void SPH::loadBitonicSort()
{

    printf("about to instantiate sorting\n");
    bitonic = Bitonic<int>( ps->cli,    
                            &cl_sort_output_hashes,
                            &cl_sort_output_indices,
                            &cl_sort_hashes,
                            &cl_sort_indices);

    /*
    //not sure i like this technique... but while the bitonic sort is still
    //using the C interface probably necessary
    static bool first_time = true;

    try {
        // if ctaSize is too large, sorting is not possible. Number of elements has to lie between some MIN 
        // and MAX array size, computed in oclRadixSort/src/RadixSort.cpp

        // SHOULD ONLY BE DONE ONCE
        if (first_time) {
            initBitonicSort(ps->cli->context(), ps->cli->queue(), 0); // no argv (last arg)
            first_time = false;
        }
    } catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        exit(0);
    }
    */
}

void SPH::bitonic_sort()
{
    try
    {
		int dir = 1; 		// dir: direction
		//int batch = num;
		int arrayLength = max_num;
        int batch = max_num / arrayLength;

        /*
		size_t szWorkgroup = bitonicSort(
                NULL,
                //d_OutputKey,
                //d_OutputVal,
				cl_sort_output_hashes.getDevicePtr(), 
				cl_sort_output_indices.getDevicePtr(), 
				cl_sort_hashes.getDevicePtr(), 
				cl_sort_indices.getDevicePtr(), 
                batch,
                arrayLength,
                dir
            );
            */

        printf("about to try sorting\n");
        bitonic.Sort(batch, arrayLength, dir);
    
	} catch (cl::Error er) {
        printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

    ps->cli->queue.finish();

	scopy(num, cl_sort_output_hashes.getDevicePtr(), 
	             cl_sort_hashes.getDevicePtr());
	scopy(num, cl_sort_output_indices.getDevicePtr(), 
	             cl_sort_indices.getDevicePtr());
    
    ps->cli->queue.finish();
}

}
