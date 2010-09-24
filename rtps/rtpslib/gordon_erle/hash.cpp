
void DataStructures::hash()
// Generate hash list: stored in cl_sort_hashes
{
	static bool first_time = true;

	ts_cl[TI_HASH]->start();

	if (first_time) {
		try {
			string path(CL_SOURCE_DIR);
			path = path + "/uniform_hash.cl";
			char* src = getSourceString(path.c_str());
			printf("before load\n");
			//printf("LOADED\n");
        	hash_program = loadProgram(src);
        	hash_kernel = cl::Kernel(hash_program, "hash", &err);
			//printf("KERNEL\n");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	cl::Kernel kern = hash_kernel;

	try {
		#if 0
		// data should already be on the GPU
    	err = queue.enqueueReadBuffer(cl_cells,  CL_TRUE, 0, nb_el*sizeof(cl_float4), &cells[0],  NULL, &event);
		queue.finish();
		#endif
	} catch(cl::Error er) {
        printf("0 ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

//  Have to make sure that the data associated with the pointers is on the GPU
	#if 0
	for (int i=0; i < 100; i++) {
		printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);
	}
	#endif


	std::vector<cl_float4>& list = cells;

	int ctaSize = 128; // work group size
	int err;


//    printf("setting up hash kernel\n");
	try {
    	err = hash_kernel.setArg(0, cl_cells);
    	err = hash_kernel.setArg(1, cl_sort_hashes);
    	err = hash_kernel.setArg(2, cl_sort_indices);
    	err = hash_kernel.setArg(3, cl_GridParams);
	} catch (cl::Error er) {
        printf("1 ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

    queue.finish();

    //printf("executing hash kernel\n");
    err = queue.enqueueNDRangeKernel(hash_kernel, cl::NullRange, cl::NDRange(nb_el), cl::NDRange(ctaSize), NULL, &event);
    queue.finish();

	//printf("int: %d, cl_int: %d\n", sizeof(int), sizeof(cl_int));
	//exit(0);


//#define DEBUG
#ifdef DEBUG
	// the kernel computes these arrays
    err = queue.enqueueReadBuffer(cl_sort_hashes,  CL_TRUE, 0, nb_el*sizeof(cl_uint), &sort_hashes[0],  NULL, &event);
    err = queue.enqueueReadBuffer(cl_sort_indices, CL_TRUE, 0, nb_el*sizeof(cl_uint), &sort_indices[0], NULL, &event);
    queue.finish();

	for (int i=0; i < 4150; i++) {  // only first 4096 are ok. WHY? 
	//for (int i=nb_el-10; i < nb_el; i++) {
		printf("sort_index: %d, sort_hash: %u, %u\n", i, sort_hashes[i], sort_indices[i]);
		printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);

		int gx = list[i].x;
		int gy = list[i].y;
		int gz = list[i].z;
		unsigned int idx = (gz*gp.grid_res.y + gy) * gp.grid_res.x + gx; 
		printf("exact hash: %d\n", idx);
		printf("---------------------------\n");
	}
#endif
#undef DEBUG

        //printf("about to read from buffers to see what they have\n");
	#if 0
		// SORT IN PLACE
        err = queue.enqueueReadBuffer(cl_sort_hashes, CL_TRUE, 0, nb_el*sizeof(cl_int), &sort_hashes[0], NULL, &event);
        err = queue.enqueueReadBuffer(cl_sort_indices, CL_TRUE, 0, nb_el*sizeof(cl_int), &sort_indices[0], NULL, &event);
		queue.finish();
		for (int i=0; i < 300; i++) {
			printf("xx index: %d, sort_indices: %d, sort_hashes: %d\n", i, sort_indices[i], sort_hashes[i]);
		}
	#endif

    queue.finish();
	ts_cl[TI_HASH]->end();
}
//----------------------------------------------------------------------
