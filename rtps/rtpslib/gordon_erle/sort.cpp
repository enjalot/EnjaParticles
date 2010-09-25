
//----------------------------------------------------------------------
// Input: a list of integers in random order
// Output: a list of sorted integers
// Leave data on the gpu
//void DataStructures::sort(BufferGE<int>& cl_hashes, BufferGE<int>& cl_indices)
void DataStructures::sort()
{
	static bool first_time = true;
	static RadixSort* radixSort;
// Sorting

	//ts_cl[TI_SORT]->start();

    try {
		// SORT IN PLACE
	#if 0
		// cl_hashes and cl_indices are correct
        err = queue.enqueueReadBuffer(cl_hashes, CL_TRUE, 0, nb_el*sizeof(cl_int), &unsort_int[0], NULL, &event);
        err = queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(cl_int), &sort_int[0], NULL, &event);
		queue.finish();
		//for (int i=0; i < nb_el; i++) {
		for (int i=0; i < 10; i++) {
			printf("*** i: %d, unsorted hash: %d, index: %d\n", i, unsort_int[i], sort_int[i]);
			//unsort_int[i] = 20000 - i;
		}
		printf("nb_el= %d\n", nb_el);
		printf("size: %d\n", sizeof(cl_int));
		exit(0);
	#endif


		// if ctaSize is too large, sorting is not possible. Number of elements has to lie between some MIN 
		// and MAX array size, computed in oclRadixSort/src/RadixSort.cpp
		int ctaSize = 64; // work group size

		// SHOULD ONLY BE DONE ONCE
		if (first_time) {
	    	radixSort = new RadixSort(context(), queue(), nb_el, "../oclRadixSort/", ctaSize, false);		    
			first_time = false;
		}

		unsigned int keybits = 32;

// **** BEFORE SORT
#if 0
	//printf("**** BEFORE SORT ******\n");
    err = queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(int), &sort_indices[0], NULL, &event);
    err = queue.enqueueReadBuffer(cl_hashes, CL_TRUE, 0, nb_el*sizeof(int), &unsort_int[0], NULL, &event);
	
    //unsort_int[0] = 2;
    //unsort_int[2] = 0;
    //sort_indices[0] = 27;
	//sort_indices[2] = 10;
    err = queue.enqueueWriteBuffer(cl_hashes, CL_TRUE, 0, nb_el*sizeof(int), &unsort_int[0], NULL, &event);
    err = queue.enqueueWriteBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(int), &sort_indices[0], NULL, &event);
    queue.finish();

    err = queue.enqueueReadBuffer(cl_hashes, CL_TRUE, 0, nb_el*sizeof(int), &unsort_int[0], NULL, &event);
    err = queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(int), &sort_indices[0], NULL, &event);
	queue.finish();

	for (int i=0; i < 10; i++) {
		// first and 3rd columns are computed by sorting method
		printf("%d: unsorted hashes: %d, sorted indices %d\n", i, unsort_int[i], sort_indices[i]);
	}
    //err = queue.enqueueWriteBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(int), &sort_indices[0], NULL, &event);
#endif

	// both arguments should already be on the GPU
	//	printf("BEFORE SORT KERNEL\n");
	    radixSort->sort(cl_sort_hashes.getDevicePtr(), cl_sort_indices.getDevicePtr(), nb_el, keybits);
	//	printf("AFTER SORT KERNEL\n");

		// Sort in place
		// NOT REQUIRED EXCEPT FOR DEBUGGING
    } catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }

	#if 0
	printf("\n**** AFTER SORT ******\n");
	queue.finish();
    std::vector<int> shash(nb_el);
    std::vector<int> sindex(nb_el);
    err = queue.enqueueReadBuffer(cl_hashes, CL_TRUE, 0, nb_el*sizeof(int), &shash[0], NULL, &event);
    err = queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, nb_el*sizeof(int), &sindex[0], NULL, &event);
	queue.finish();
	for (int i=0; i < nb_el; i++) {
		// first and 3rd columns are computed by sorting method
		printf("%d: sorted hash: %d, sorted index; %d\n", i, shash[i], sindex[i]);
	}
	#endif

    ps->cli->queue.finish();
	//ts_cl[TI_SORT]->end();
}
//----------------------------------------------------------------------
