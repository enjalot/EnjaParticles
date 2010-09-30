
#include <string>
#include "oclSortingNetworks_common.h"
#include "GE_SPH.h"
#include "BufferGE.h"

#if 1
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
#endif

using namespace std;

namespace rtps {

//----------------------------------------------------------------------
// Input: a list of integers in random order
// Output: a list of sorted integers
// Leave data on the gpu

void GE_SPH::bitonic_sort()
{
	static bool first_time = true;
	int ctaSize = 64; // work group size

	//printf("ENTER BISORT\n");

	ts_cl[TI_SORT]->start();

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

	//return;
	unsigned int keybits = 32;

	BufferGE<int> cl_sort_output_hashes(ps->cli, nb_el);
	BufferGE<int> cl_sort_output_indices(ps->cli, nb_el);

	try {
	#if 0
		//prepareSortData();
		printf("nb_el= %d\n", nb_el);
		cl_sort_hashes->copyToHost();
		cl_sort_indices->copyToHost();
		for (int i=0; i < nb_el; i++) {
			printf("** hash: %d, index: %d\n", (*cl_sort_hashes)[i], (*cl_sort_indices)[i]);
		}
	#endif



	//printf("nb_el= %d\n", nb_el); exit(0);
	// both arguments should already be on the GPU
		// CRASHES EVERY 2-3 runs. WHY? 

		int dir = 1; 		// dir: direction
		int batch = nb_el;

		#if 1
		size_t szWorkgroup = bitonicSort(
                NULL,
                //d_OutputKey,
                //d_OutputVal,
				cl_sort_output_hashes.getDevicePtr(), 
				cl_sort_output_indices.getDevicePtr(), 
				cl_sort_hashes->getDevicePtr(), 
				cl_sort_indices->getDevicePtr(), 
                //d_InputKey,
                //d_InputVal,
                //nb_el / arrayLength,
                //arrayLength,
                nb_el / batch,
				batch,
                dir
            );
		#endif
	} catch (cl::Error er) {
        printf("ERROR(radixSort->sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	#if 1
	cl_sort_output_hashes.copyToHost();
	cl_sort_output_indices.copyToHost();

	int* sh = cl_sort_hashes->getHostPtr();
	int* si = cl_sort_indices->getHostPtr();
	int* soh = cl_sort_output_hashes.getHostPtr();
	int* soi = cl_sort_output_indices.getHostPtr();

	for (int i=0; i < nb_el; i++) {
		//cl_sort_hashes[i] = cl_sort_output_hashes[i];
		//cl_sort_indices[i] = cl_sort_output_indices[i];
		sh[i] = soh[i];
		si[i] = soi[i];
	}

	cl_sort_hashes->copyToDevice();
	cl_sort_indices->copyToDevice();
	#endif

    ps->cli->queue.finish();
	ts_cl[TI_SORT]->end();

	printf("enter sort diagonistics ****\n");

	printBiSortDiagnostics(cl_sort_output_hashes, cl_sort_output_indices);

	printf("EXIT BISORT \n");
}
//----------------------------------------------------------------------
void GE_SPH::printBiSortDiagnostics(BufferGE<int>& cl_sort_output_hashes, BufferGE<int>& cl_sort_output_indices)
{
#if 1
    cl_sort_hashes->copyToHost(); 
    cl_sort_indices->copyToHost(); 
    cl_sort_output_hashes.copyToHost(); 
    cl_sort_output_indices.copyToHost(); 
	ps->cli->queue.finish();

	int* hashi = cl_sort_hashes->getHostPtr();
	int* sorti = cl_sort_indices->getHostPtr();

	int* ohashi = cl_sort_output_hashes.getHostPtr();
	int* osorti = cl_sort_output_indices.getHostPtr();

	for (int i=0; i < nb_el; i++) {
	//for (int i=0; i < 200; i++) {
		printf("=========================================\n");
		printf("sorted hash[%d]: %d, sorted index[%d]: %d\n", i, hashi[i], i, sorti[i]);
		printf("osorted hash[%d]: %d, osorted indx[%d]: %d\n", i, ohashi[i], i, osorti[i]);
	}
#endif
}
//----------------------------------------------------------------------
void GE_SPH::prepareSortData()
{
#if 1
	printf("**** BEFORE SORT ****\n");
		// cl_hashes and cl_indices are correct
        cl_sort_hashes->copyToHost(); 
        cl_sort_indices->copyToHost(); 
		ps->cli->queue.finish();

		int* hashi = cl_sort_hashes->getHostPtr();
		int* sorti = cl_sort_indices->getHostPtr();

		for (int i=0; i < nb_el; i++) {
		//for (int i=0; i < 200; i++) {
			printf("*** i: %d, unsorted hash: %d, index: %d\n", i, hashi[i], sorti[i]);
		}
#endif
}
//----------------------------------------------------------------------

} // namespace
