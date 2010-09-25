
void DataStructures::hash()
// Generate hash list: stored in cl_sort_hashes
{
	static bool first_time = true;

	//ts_cl[TI_HASH]->start();

	if (first_time) {
		try {
			string path(CL_SOURCE_DIR);
			path = path + "/uniform_hash.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);

        	hash_kernel = Kernel(ps->cli, strg, "hash");
			//printf("KERNEL\n");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = hash_kernel;

//  Have to make sure that the data associated with the pointers is on the GPU
	#if 0
	for (int i=0; i < 100; i++) {
		printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);
	}
	#endif



	int ctaSize = 128; // work group size
	int err;


	kern.setArg(0, cl_cells);
	kern.setArg(1, cl_sort_hashes);
	kern.setArg(2, cl_sort_indices);
	kern.setArg(3, cl_GridParams);

    //printf("executing hash kernel\n");
	kern.execute(nb_el,ctaSize);


//#define DEBUG
#ifdef DEBUG
	// the kernel computes these arrays
	cl_sort_hashes.getDevicePtr();
	cl_sort_indices.getDevicePtr();

	for (int i=0; i < 4150; i++) {  // only first 4096 are ok. WHY? 
	//for (int i=nb_el-10; i < nb_el; i++) {
		printf("sort_index: %d, sort_hash: %u, %u\n", i, sort_hashes[i], sort_indices[i]);
		printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);

		#if 0
		int gx = list[i].x;
		int gy = list[i].y;
		int gz = list[i].z;
		unsigned int idx = (gz*gp.grid_res.y + gy) * gp.grid_res.x + gx; 
		printf("exact hash: %d\n", idx);
		printf("---------------------------\n");
		#endif
	}
#endif
#undef DEBUG

        //printf("about to read from buffers to see what they have\n");
	#if 0
		// SORT IN PLACE
		cl_sort_hashes.getDevices();
		cl_sort_indices.getDevices();

		ps->cli->queue.finish();
		for (int i=0; i < 300; i++) {
			printf("xx index: %d, sort_indices: %d, sort_hashes: %d\n", i, sort_indices[i], sort_hashes[i]);
		}
	#endif

	//ts_cl[TI_HASH]->end();
}
//----------------------------------------------------------------------
