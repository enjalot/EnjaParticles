

//----------------------------------------------------------------------
void DataStructures::buildDataStructures()
{
	static bool first_time = true;

	// nb_vars: number of float4 variables to reorder. 
	// nb_el:   number of particles
	// Alternative: could construct float columns
	// Stored in vars_sorted[nb_vars*nb_el]. Ordering is consistent 
	// with vars_sorted[nb_vars][nb_el]

	//ts_cl[TI_BUILD]->start();

	CL cl;

	if (first_time) {
		try {
			string path(CL_SOURCE_DIR);
			path = path + "/datastructures_test.cl";
			//char* src = getSourceString(path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);

        	datastructures_program = cl.loadProgram(std::string(src));
        	datastructures_kernel = cl::Kernel(datastructures_program, "datastructures", &err);
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	cl::Kernel kern = datastructures_kernel;

	int ctaSize = 128;

	try {
    	err = datastructures_kernel.setArg(0, nb_el);
    	err = datastructures_kernel.setArg(1, nb_vars);
    	err = datastructures_kernel.setArg(2, cl_vars_unsorted);
    	err = datastructures_kernel.setArg(3, cl_vars_sorted);
    	err = datastructures_kernel.setArg(4, cl_sort_hashes);
    	err = datastructures_kernel.setArg(5, cl_sort_indices);
    	err = datastructures_kernel.setArg(6, cl_cell_indices_start);
    	err = datastructures_kernel.setArg(7, cl_cell_indices_end);
		// local memory
    	err = datastructures_kernel.setArg(8, (ctaSize+1)*sizeof(int), 0);
	} catch(cl::Error er) {
        printf("0 ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}


	int err;

	try {
    	err = queue.enqueueNDRangeKernel(datastructures_kernel, cl::NullRange, cl::NDRange(nb_el), cl::NDRange(ctaSize), NULL, &event);
		queue.finish();
	} catch(cl::Error er) {
        printf("exec: ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	int nb_bytes;

#if 1
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
#endif


	#if 0
	// PRINT OUT UNSORTED AND SORTED VARIABLES
	for (int k=0; k < nb_vars; k++) {
		printf("================================================\n");
		printf("=== VARIABLE: %d ===============================\n", k);
	for (int i=0; i < nb_el; i++) {
		cl_float4 us = vars_unsorted[i+k*nb_el];
		cl_float4 so = vars_sorted[i+k*nb_el];
		printf("[%d]: %d, (%f, %f), (%f, %f)\n", i, sort_indices[i], us.x, us.y, so.x, so.y);
	}
	}
	printf("===============================================\n");
	printf("===============================================\n");
	#endif


	#if 0
	try {
		// PRINT OUT START and END CELL INDICES
		nb_bytes = grid_size*sizeof(cl_int);
        err = queue.enqueueReadBuffer(cl_cell_indices_start, CL_TRUE, 0, nb_bytes, &cell_indices_start[0], NULL, &event);
        err = queue.enqueueReadBuffer(cl_cell_indices_end, CL_TRUE, 0, nb_bytes, &cell_indices_end[0], NULL, &event);
	} catch(cl::Error er) {
        printf("2 ERROR(buildDatastructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

		printf("cell_indices_start, end\n");
		int nb_cells = 0;
		for (int i=0; i < grid_size; i++) {
			int nb = cell_indices_end[i]-cell_indices_start[i];
			nb_cells += nb;
			printf("[%d]: %d, %d, nb pts: %d\n", i, cell_indices_start[i], cell_indices_end[i], nb);
		}
		printf("total nb cells: %d\n", nb_cells);
	#endif

	//printf("return from BuildDataStructures\n");

    queue.finish();
	//ts_cl[TI_BUILD]->end();
}
//----------------------------------------------------------------------
