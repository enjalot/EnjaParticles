
void DataStructures::setupArrays()
{
	// only for my test routines: sort, hash, datastructures
	int nb_bytes;

	nb_el = (1 << 16);  // number of particles
	printf("nb_el= %d\n", nb_el); //exit(0);
	nb_vars = 3;        // number of cl_float4 variables to reorder
	printf("nb_el= %d\n", nb_el); 

printf("\n\nBEFORE cell BufferGE<float4>\n"); //********************
	cl_cells = BufferGE<float4>(ps->cli, nb_el);
printf("AFTER cell BufferGE\n");
	// notice the index rotation? 

	for (int i=0; i < nb_el; i++) {
		float aa = rand_float(0.,10.);
		cl_cells[i].x = rand_float(0.,10.);
		cl_cells[i].y = rand_float(0.,10.);
		cl_cells[i].z = rand_float(0.,10.);
		cl_cells[i].w = 1.;
		//printf("%d, %f, %f, %f, %f\n", i, cells[i].x, cells[i].y, cells[i].z, cells[i].w);
	}
	cl_cells.copyToDevice();

printf("\n\nBEFORE BufferGE<GridParams> check\n"); //**********************
// Need an assign operator (no memory allocation)
	cl_GridParams = BufferGE<GridParams>(ps->cli, 1); // destroys ...

	GridParams& gp = *(cl_GridParams.getHostPtr());
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
	cl_GridParams.copyToDevice();
//exit(0);

	printf("delta z= %f\n", gp.grid_delta.z);

	grid_size = (int) (gp.grid_res.x * gp.grid_res.y * gp.grid_res.z);
	printf("grid_size= %d\n", grid_size);

	cl_unsort = BufferGE<int>(ps->cli, nb_el);
	cl_sort   = BufferGE<int>(ps->cli, nb_el);

	int* iunsort = cl_unsort.getHostPtr();
	int* isort = cl_sort.getHostPtr();

	for (int i=0; i < nb_el; i++) {
		isort[i] = i;
		iunsort[i] = nb_el-i;
		//cl_sort[i] = i;  // DOES NOT WORK, but works with cl_cells
		//cl_unsort[i] = nb_el-i;
	}


	// position POS=0
	// velocity VEL=1
	// Force    FOR=2
	cl_vars_unsorted = BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_vars_sorted = BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_cell_indices_start = BufferGE<int>(ps->cli, nb_el);
	cl_cell_indices_end   = BufferGE<int>(ps->cli, nb_el);
	cl_sort_indices = BufferGE<int>(ps->cli, nb_el);
	cl_sort_hashes = BufferGE<int>(ps->cli, nb_el);
    printf("about to start writing buffers\n");

	#if 0
	cl_vars_unsorted.copyToDevice();
	cl_vars_sorted.copyToDevice();
	cl_cell_indices_start.copyToDevice();
	cl_cell_indices_end.copyToDevice();
	cl_sort_indices.copyToDevice();
	cl_sort_hashes.copyToDevice();
	#endif

	int nb_floats = nb_vars*nb_el;
	// WHY NOT cl_float4 (in CL/cl_platform.h)
	float4 f;
	float4 zero;

	zero.x = zero.y = zero.z = 0.0;
	zero.w = 1.;

	for (int i=0; i < nb_floats; i++) {
		f.x = rand_float(0., 1.);
		f.y = rand_float(0., 1.);
		f.z = rand_float(0., 1.);
		f.w = 1.0;
		cl_vars_unsorted[i] = f;
		cl_vars_sorted[i]   = zero;
		//printf("f= %f, %f, %f, %f\n", f.x, f.y, f.z, f.w);
	}

	// SETUP FLUID PARAMETERS
	// cell width is one diameter of particle, which imlies 27 neighbor searches
	cl_FluidParams = BufferGE<FluidParams>(ps->cli, 1);
	FluidParams& fp = *cl_FluidParams.getHostPtr();;
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
	cl_FluidParams.copyToDevice();



    printf("done with setup arrays\n");
}
//----------------------------------------------------------------------
