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
<<<<<<< HEAD
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
=======

    printf("about to instantiate sorting\n");
    
    bitonic = Bitonic<int>( ps->cli,    
                            &cl_sort_output_hashes,
                            &cl_sort_output_indices,
                            &cl_sort_hashes,
                            &cl_sort_indices);
    
>>>>>>> 0c48835a5c20dfa6e1fcb2ca9211f679aabe5a26
}

void SPH::bitonic_sort(bool ghosts)
{
    try
    {
		int dir = 1; 		// dir: direction
		//int batch = num;
        if(ghosts)
        {
            int arrayLength = nb_ghosts;
            int batch = nb_ghosts / arrayLength;

            ghost_bitonic.Sort(batch, arrayLength, dir);

        }
        else
        {

		    int arrayLength = max_num;
            int batch = max_num / arrayLength;

            //printf("about to try sorting\n");
            bitonic.Sort(batch, arrayLength, dir);
        }
<<<<<<< HEAD
=======
    
>>>>>>> 0c48835a5c20dfa6e1fcb2ca9211f679aabe5a26
	} catch (cl::Error er) {
        printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

    ps->cli->queue.finish();

    if(ghosts)
    {
        printf("ghosts scopy\n");
        scopy(num, cl_ghosts_sort_output_hashes.getDevicePtr(), 
                     cl_ghosts_sort_hashes.getDevicePtr());
        scopy(num, cl_ghosts_sort_output_indices.getDevicePtr(), 
                     cl_ghosts_sort_indices.getDevicePtr());

        printf("ghosts done scopy\n");
 
    }
    else
    {
        scopy(num, cl_sort_output_hashes.getDevicePtr(), 
                     cl_sort_hashes.getDevicePtr());
        scopy(num, cl_sort_output_indices.getDevicePtr(), 
                     cl_sort_indices.getDevicePtr());
    }
    
<<<<<<< HEAD
=======
    /*
    int nbc = 10;
    std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
    std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    for(int i = 0; i < nbc; i++)
    {
        printf("before[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
    }
    printf("\n");
    */


	/*
>>>>>>> 0c48835a5c20dfa6e1fcb2ca9211f679aabe5a26
    ps->cli->queue.finish();
}

}
