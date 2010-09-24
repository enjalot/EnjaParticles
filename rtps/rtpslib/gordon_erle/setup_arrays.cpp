
void DataStructures::setupArrays()
{
	// only for my test routines: sort, hash, datastructures
	int nb_bytes;

	nb_el = (1 << 16);  // number of particles
	printf("nb_el= %d\n", nb_el); //exit(0);
	nb_vars = 3;        // number of cl_float4 variables to reorder
	printf("nb_el= %d\n", nb_el); 

	cells.resize(nb_el);
	// notice the index rotation? 

	for (int i=0; i < nb_el; i++) {
		cells[i].x = rand_float(0.,10.);
		cells[i].y = rand_float(0.,10.);
		cells[i].z = rand_float(0.,10.);
		cells[i].w = 1.;
		//printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);
	}

	GridParams gp;
	//float resol = 50.;
	float resol = 25.;
	gp.grid_size = float4(10.,10.,10.,1.);
	gp.grid_min  = float4(0.,0.,0.,1.);
	gp.grid_max.x = gp.grid_size.x + gp.grid_min.x; 
	gp.grid_max.y = gp.grid_size.y + gp.grid_min.y; 
	gp.grid_max.z = gp.grid_size.z + gp.grid_min.z; 
	gp.grid_max.w = 1.0;
	gp.grid_res  = float4(resol,resol,resol,1.);
	gp.grid_delta.x = gp.grid_size.x / gp.grid_res.x;
	gp.grid_delta.y = gp.grid_size.y / gp.grid_res.y;
	gp.grid_delta.z = gp.grid_size.z / gp.grid_res.z;
	// I want inverse of delta to use multiplication in the kernel
	gp.grid_delta.x = 1. / gp.grid_delta.x;
	gp.grid_delta.y = 1. / gp.grid_delta.y;
	gp.grid_delta.z = 1. / gp.grid_delta.z;
	gp.grid_delta.w = 1.;
	gp.numParticles = nb_el;  // WRONG SPOT, BUT USEFUL for CL kernels arg passing
	std::vector<GridParams> vgp(1);
	vgp.push_back(gp);
	cl_GridParams = Buffer<GridParams>(ps->cli, vgp);

	printf("delta z= %f\n", gp.grid_delta.z);

	grid_size = (int) (gp.grid_res.x * gp.grid_res.y * gp.grid_res.z);
	printf("grid_size= %d\n", grid_size);

	sort_int.resize(nb_el);
	unsort_int.resize(nb_el);

	for (int i=0; i < nb_el; i++) {
		sort_int.push_back(i);
		unsort_int.push_back(nb_el-i);
	}

	cl_unsort = Buffer<int>(ps->cli, unsort);
	cl_sort   = Buffer<int>(ps->cli, sort);

	// position POS=0
	// velocity VEL=1
	// Force    FOR=2
	vars_unsorted.resize(nb_el*nb_vars);
	vars_sorted.resize(nb_el*nb_vars);
	cell_indices_start.resize(nb_el);
	cell_indices_end.resize(nb_el);
	sort_indices.resize(nb_el);
	sort_hashes.resize(nb_el);
    printf("about to start writing buffers\n");

	int nb_floats = nb_vars*nb_el;
	cl_float4 f;
	cl_float4 zero;
	zero.x = zero.y = zero.z = 0.0;
	zero.w = 1.;

	for (int i=0; i < nb_floats; i++) {
		f.x = rand_float(0., 1.);
		f.y = rand_float(0., 1.);
		f.z = rand_float(0., 1.);
		f.w = 1.0;
		vars_unsorted[i] = f;
		vars_sorted[i]   = zero;
		//printf("f= %f, %f, %f, %f\n", f.x, f.y, f.z, f.w);
	}

	// SETUP FLUID PARAMETERS
	// cell width is one diameter of particle, which imlies 27 neighbor searches
	float radius = gp.grid_delta.x; 
	fp.smoothing_length = radius; // SPH radius
	fp.scale_to_simulation = 1.0; // overall scaling factor
	fp.mass = 1.0; // mass of single particle (MIGHT HAVE TO BE CHANGED)
	fp.friction_coef = 0.1;
	fp.restitution_coef = 0.9;
	fp.damping = 0.9;
	fp.shear = 0.9;
	fp.attraction = 0.9;
	fp.spring = 0.5;
	fp.gravity = -9.8; // -9.8 m/sec^2


#define BUFFER(bytes) cl::Buffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
#define WRITE_BUFFER(cl_var, bytes, cpu_var_ptr) queue.enqueueWriteBuffer(cl_var, CL_TRUE, 0, bytes, cpu_var_ptr, NULL, &event)

	try {

		cl_vars_unsorted = Buffer(ps->cli, vars_unsorted);
		cl_vars_unsorted->copyToDevice();
		//----------------
		// float4 ELEMENTS
		nb_bytes = nb_el*nb_vars*sizeof(cl_float4);

		cl_vars_unsorted = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_vars_unsorted, nb_bytes, &vars_unsorted[0]);

		cl_vars_sorted = BUFFER(nb_bytes); 
		WRITE_BUFFER(cl_vars_sorted, nb_bytes, &vars_sorted[0]);

		nb_bytes = nb_el*sizeof(cl_float4);
		cl_cells = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_cells, nb_bytes, &cells[0]);

		//----------------
		// int ELEMENTS
		nb_bytes = nb_el*sizeof(int);
		cl_sort_hashes  = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_sort_hashes, nb_bytes, &sort_hashes[0]);

		cl_sort_indices = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_sort_indices, nb_bytes, &sort_indices[0]);

		cl_unsort = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_unsort, nb_bytes, &unsort_int);

		cl_sort = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_sort, nb_bytes, &sort_int);

		nb_bytes = sizeof(GridParams);
		cl_GridParams = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_GridParams, nb_bytes, &gp);

		nb_bytes = sizeof(FluidParams);
		cl_FluidParams = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_FluidParams, nb_bytes, &fp);

		nb_bytes = grid_size * sizeof(int);
		cl_cell_indices_start = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_cell_indices_start, nb_bytes, &cell_indices_start[0]);

		cl_cell_indices_end = BUFFER(nb_bytes);
		WRITE_BUFFER(cl_cell_indices_end, nb_bytes, &cell_indices_end[0]);

		queue.finish();
	} catch(cl::Error er) {
        printf("ERROR(setupArray): %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
	}

	//exit(0);


    printf("done with setup arrays\n");
#undef BUFFER
#undef WRITE_BUFFER
}
//----------------------------------------------------------------------
