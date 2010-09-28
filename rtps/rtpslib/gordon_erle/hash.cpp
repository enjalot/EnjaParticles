
void DataStructures::hash()
// Generate hash list: stored in cl_sort_hashes
{
	static bool first_time = true;

	ts_cl[TI_HASH]->start();

	if (first_time) {
		try {
			string path(CL_UTIL_SOURCE_DIR);
			path = path + "/uniform_hash.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);

        	hash_kernel = Kernel(ps->cli, strg, "hash");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(hash): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = hash_kernel;

	float4* cells = cl_cells.getHostPtr();
	int* sort_hashes  = cl_sort_hashes.getHostPtr();
	int* sort_indices = cl_sort_indices.getHostPtr();

	int ctaSize = 128; // work group size
	int err;

	kern.setArg(0, cl_cells.getDevicePtr());
	kern.setArg(1, cl_sort_hashes.getDevicePtr());
	kern.setArg(2, cl_sort_indices.getDevicePtr());
	kern.setArg(3, cl_GridParams.getDevicePtr());

    //printf("executing hash kernel\n");
	printf("nb_el= %d\n", nb_el);
	kern.execute(nb_el,ctaSize);

	ps->cli->queue.finish();
	ts_cl[TI_HASH]->end();
}
//----------------------------------------------------------------------
void DataStructures::printHashDiagnostics()
{
#if 0
	cl_GridParams.copyToHost();
	GridParams& gp = *cl_GridParams.getHostPtr();
	printf("%f, %f, %f\n", gp.grid_res.x, gp.grid_res.y, gp.grid_res.z);
#endif

//#define DEBUG
#ifdef DEBUG
	cl_sort_hashes.copyToHost();
	cl_sort_indices.copyToHost();
	for (int i=0; i < nb_el; i++) {  // only first 4096 are ok. WHY? 
		printf(" sort_hash[%d] %u, sort_indices[%d]: %u\n", i, sort_hashes[i], i, sort_indices[i]);
		//printf("cells[%d], %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);

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
}
//----------------------------------------------------------------------
