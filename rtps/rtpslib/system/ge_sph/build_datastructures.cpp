

//----------------------------------------------------------------------
void DataStructures::buildDataStructures()
{
	static bool first_time = true;

	ts_cl[TI_BUILD]->start();

	if (first_time) {
		try {
			string path(CL_UTIL_SOURCE_DIR);
			path = path + "/datastructures_test.cl";
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	datastructures_kernel = Kernel(ps->cli, strg, "datastructures");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = datastructures_kernel;

	int workSize = 128;

	// HOW TO DEAL WITH ARGUMENTS

	kern.setArg(0, nb_el);
	kern.setArg(1, nb_vars);
	kern.setArg(2, cl_vars_unsorted->getDevicePtr());
	kern.setArg(3, cl_vars_sorted->getDevicePtr());
	kern.setArg(4, cl_sort_hashes->getDevicePtr());
	kern.setArg(5, cl_sort_indices->getDevicePtr());
	kern.setArg(6, cl_cell_indices_start->getDevicePtr());
	kern.setArg(7, cl_cell_indices_end->getDevicePtr());

	// local memory
	int nb_bytes = (workSize+1)*sizeof(int);
    kern.setArgShared(8, nb_bytes);

	int err;
   	kern.execute(nb_el, workSize); 

	printBuildDiagnostics();

    ps->cli->queue.finish();
	ts_cl[TI_BUILD]->end();
}
//----------------------------------------------------------------------
void DataStructures::printBuildDiagnostics()
{
#if 0
	// should try with and without exceptions. 
	// DATA SHOULD STAY ON THE GPU
	try {
		nb_bytes = nb_el*nb_vars*sizeof(cl_float4);
		cl_vars_unsorted.copyToDevice();
		cl_vars_sorted.copyToDevice();
		cl_vars_sort_indices.copyToDevice();
	} catch(cl::Error er) {
        printf("1 ERROR(buildDatastructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	// PRINT OUT UNSORTED AND SORTED VARIABLES
	for (int k=0; k < nb_vars; k++) {
		printf("================================================\n");
		printf("=== VARIABLE: %d ===============================\n", k);
	for (int i=0; i < nb_el; i++) {
		float4 us = cl_vars_unsorted[i+k*nb_el];
		float4 so = cl_vars_sorted[i+k*nb_el];
		printf("[%d]: %d, (%f, %f), (%f, %f)\n", i, cl_sort_indices[i], us.x, us.y, so.x, so.y);
	}
	}
	printf("===============================================\n");
	printf("===============================================\n");


	try {
		// PRINT OUT START and END CELL INDICES
		cl_cell_indices_start.copyToHost();
		cl_cell_indices_end.copyToHost();
	} catch(cl::Error er) {
        printf("2 ERROR(buildDatastructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

		printf("cell_indices_start, end\n");
		int nb_cells = 0;
		for (int i=0; i < grid_size; i++) {
			int nb = cl_cell_indices_end[i]-cl_cell_indices_start[i];
			nb_cells += nb;
			printf("[%d]: %d, %d, nb pts: %d\n", i, cl_cell_indices_start[i], cl_cell_indices_end[i], nb);
		}
		printf("total nb cells: %d\n", nb_cells);
#endif

	//printf("return from BuildDataStructures\n");
}
//----------------------------------------------------------------------
