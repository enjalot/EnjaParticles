#include "SPH.h"
#include "oclSortingNetworks_common.h"

#include <string.h>

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

namespace rtps
{

void SPH::loadBitonicSort()
{
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
}

void SPH::bitonic_sort()
{
    try
    {
		int dir = 1; 		// dir: direction
		int batch = num;

		size_t szWorkgroup = bitonicSort(
                NULL,
                //d_OutputKey,
                //d_OutputVal,
				cl_sort_output_hashes.getDevicePtr(), 
				cl_sort_output_indices.getDevicePtr(), 
				cl_sort_hashes.getDevicePtr(), 
				cl_sort_indices.getDevicePtr(), 
                //d_InputKey,
                //d_InputVal,
                //nb_el / arrayLength,
                //arrayLength,
                num / batch,
				batch,
                dir
            );
	} catch (cl::Error er) {
        printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

    ps->cli->queue.finish();

    /*
	scopy(num, cl_sort_output_hashes->getDevicePtr(), 
	             cl_sort_hashes->getDevicePtr());
	scopy(num, cl_sort_output_indices->getDevicePtr(), 
	             cl_sort_indices->getDevicePtr());
    */
    
}

}
