#include "GE_SPH.h"
#include "RadixSort.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Input: a list of integers in random order
// Output: a list of sorted integers
// Leave data on the gpu

void GE_SPH::sort()
{
	//exit(0);
	static bool first_time = true;
	int ctaSize = 64; // work group size

	//printf("ENTER SORT\n");

	ts_cl[TI_SORT]->start();

    try {
		// if ctaSize is too large, sorting is not possible. Number of elements has to lie between some MIN 
		// and MAX array size, computed in oclRadixSort/src/RadixSort.cpp

		// SHOULD ONLY BE DONE ONCE
		if (first_time) {
			printf("radixSort: nb_el= %d\n", nb_el);
	    	radixSort = new RadixSort(ps->cli->context(), ps->cli->queue(), nb_el, "../oooclRadixSort/", ctaSize, false);		    
			first_time = false;
		}
    } catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }

	//return;
	unsigned int keybits = 32;

	try {
		//prepareSortData();
		#if 0
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
   		radixSort->sort(cl_sort_hashes->getDevicePtr(), 
			cl_sort_indices->getDevicePtr(), nb_el, keybits);
	} catch (cl::Error er) {
        printf("ERROR(radixSort->sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

    ps->cli->queue.finish();
	ts_cl[TI_SORT]->end();

	//printf("enter sort diagonistics ****\n");
	//printSortDiagnostics();

	//printf("EXIT SORT \n");
}
//----------------------------------------------------------------------
void GE_SPH::printSortDiagnostics()
{
#if 1
    cl_sort_hashes->copyToHost(); 
    cl_sort_indices->copyToHost(); 
	ps->cli->queue.finish();

	int* hashi = cl_sort_hashes->getHostPtr();
	int* sorti = cl_sort_indices->getHostPtr();

	for (int i=0; i < nb_el; i++) {
	//for (int i=0; i < 200; i++) {
		printf("=========================================\n");
		printf("sorted hash[%d]: %d, sorted index[%d]: %d\n", i, hashi[i], i, sorti[i]);
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
