
#include <GL/glew.h>
#include <math.h>

#include "GE_SPH.h"
#include "../particle/UniformGrid.h"

// GE: need it have access to my datastructure (GE). Will remove 
// eventually. 
//#include "datastructures.h"

namespace rtps {


//----------------------------------------------------------------------
//void GE_SPH::setGEDataStructures(DataStructures* ds)
//{
	//this->ds = ds;
//}
//----------------------------------------------------------------------
GE_SPH::GE_SPH(RTPS *psfr, int n)
{
    //for reading back different values from the kernel
    std::vector<float4> error_check(num);

    //store the particle system framework
    ps = psfr;

	radixSort = 0;

    num = n;
	// density, force, pos, vel, surf tension, color
	nb_vars = 7;  // for array structure in OpenCL
	nb_el = n;

	// STRATEGY
	// mass of single particle based on density and particle radius
	// mass of single cell of the grid, assuming np particles per cell
	//   (np is 1 or 8 in 3D)

	// Density (mass per unit vol)
	// density of water: 1000 kg/m^3  (62lb/ft^3)
	double density = 1000.; // water: 62 lb/ft^3 = 1000 kg/m^3

	double pi = 3.14159;
	double nb_particles = num;
	double particle_radius = 0.004;
	double particle_volume = (4.*pi/3.) * (particle_radius, 3.);
	double particle_mass = particle_volume * density;
	particle_mass = 0.00020543; // from Fluids2
	particle_volume = particle_mass / density;
	particle_radius = pow(particle_volume*3./(4.*pi), 1./3.);
	printf("particle radius= %f\n", particle_radius);

	int nb_particles_in_cell = 1;
	// mass of fluid in a single cell
	float cell_mass = nb_particles_in_cell*particle_mass;
	float cell_volume = cell_mass / density; // = particle_volume

	// Cell contains nb_particles_in_cell of fluid
	// size of single fluid element viewed as a cube

	// particle "cube" with same mass as spherical particle
	float particle_size = pow(cell_volume, 1./3.);

	float particle_spacing;
	// particle_spacing of particles at t=0
	// rest distance between particles
	float particle_rest_distance = 0.87*particle_size; // why 0.87? 
	particle_spacing = particle_rest_distance; 

	printf("particle_spacing= %f\n", particle_spacing);
	printf("particle_rest_distance= %f\n", particle_rest_distance);

	#if 0
	if (nb_particles_in_cell == 1) {
		particle_spacing = cell_sz;
	} else if (nb_particles_in_cell > 1) {
		particle_spacing = pow((float) nb_particles_in_cell, -1./3.) * cell_sz;
	} else {
		printf("nb_particles_in_cell must be >= 1\n");
		exit(0);
	}
	#endif

	// desired number of particles within the smoothing_distance sphere
	int nb_interact_part = 50;
	// (h/radius)^3 = nb_interact_part
	double h = pow(nb_interact_part, 1./3.) * particle_radius;
	//h = .02; // larger than I would want it, but same as Fluid v.2
	//h = .01; // larger than I would want it, but same as Fluid v.2
	// domain cell size
	float cell_sz;
	cell_sz = h;  // 27 neighbor search (only neighbors strictly necessary)
	printf("cell_size, delta_x= %f\n", cell_sz);
	printf("h= %f\n", h);

	//-------------------------------
	//SETUP GRID

    sph_settings.simulation_scale = 0.01; //0.004;

	double cell_size = cell_sz;
	printf("cell_size= %f\n", cell_size); 

	//cell_size = h;
	int    nb_cells_x; // number of cells along x
	int    nb_cells_y; 
	int    nb_cells_z;

	// Dam (repeat case from Fluids v2
	// Size in world space
	float4 domain_min = float4(-10., -5., 0., 1.);
	float4 domain_max = float4(+10., +5., 10., 1.);
	float4 fluid_min   = float4(-0.5, -4.9,  0.1, 1.);
	float4 fluid_max   = float4( 9.9, +4.9,  7., 1.);

	double domain_size_x = domain_max.x - domain_min.x; 
	double domain_size_y = domain_max.y - domain_min.y; 
	double domain_size_z = domain_max.z - domain_min.z; 

	float world_cell_size = cell_size / sph_settings.simulation_scale;
	printf("world_cell_size= %f\n", world_cell_size);

	nb_cells_x = (int) (domain_size_x / world_cell_size);
	nb_cells_y = (int) (domain_size_y / world_cell_size);
	nb_cells_z = (int) (domain_size_z / world_cell_size);

	printf("nb cells: %d, %d, %d\n", nb_cells_x, nb_cells_y, nb_cells_z);
	printf("part_rest_world: %f\n", particle_spacing / sph_settings.simulation_scale);

	//-------------------------------
    
    //init sph stuff
    sph_settings.rest_density = density;
    sph_settings.particle_mass = particle_mass;

	// Do not know why required  REST DISTANCE
    sph_settings.particle_rest_distance = particle_rest_distance; 
    sph_settings.particle_spacing = particle_spacing;  // distance between particles
    sph_settings.smoothing_distance = h;   // CHECK THIS. Width of W function
    sph_settings.particle_radius = particle_radius; 

    sph_settings.boundary_distance = sph_settings.particle_spacing / 2.;

	printf("domain size: %f, %f, %f\n", domain_size_x, domain_size_y, domain_size_z);
	//exit(0);

	//float4 domain_size(domain_size_x, domain_size_y, domain_size_z, 1.);
	float4 domain_origin = domain_min;
	float4 domain_size;
	domain_size.x = domain_size_x;
	domain_size.y = domain_size_y;
	domain_size.z = domain_size_z;
	domain_size.w = 1.0;
	domain_origin.print("domain origin");
	domain_size.print("domain size");


	nb_cells = int4(nb_cells_x, nb_cells_y, nb_cells_z, 1);
	int num_old = num;
    grid = UniformGrid(domain_min, domain_max, nb_cells, sph_settings.simulation_scale); 

	printf("simu scale: %f\n", sph_settings.simulation_scale);
	printf("UniformGrid\n");
	domain_origin.print("origin");
	domain_size.print("domain size");

	//END SETUP GRID
	//-------------------------------


printf("num= %d\n", num);

    //*** Initialization, TODO: move out of here to the particle directory
    positions.resize(num);
    forces.resize(num);
    velocities.resize(num);
    densities.resize(num);

	// CREATE FLUID SPHERE
	float4 center(5., 5., 8., 1.);
	float radius = 0.5;
	int offset = 0;
	printf("original offset: %d\n", offset);
	//grid.makeSphere(&positions[0], center, radius, num, offset, 
	//	sph_settings.particle_spacing);
	printf("after sphere, offset: %d\n", offset);


	float4 pmin = fluid_min;
	float4 pmax = fluid_max;
	pmin.print("pmin");
	pmax.print("pmax");
	printf("particle_spacing: %f\n", sph_settings.particle_spacing);

	// INITIATE PARTICLE POSITIONS
	float spacing = sph_settings.particle_spacing/sph_settings.simulation_scale;
	printf("spacing= %f\n", spacing);

	grid.makeCube(&positions[0], pmin, pmax, sph_settings.particle_spacing/sph_settings.simulation_scale, num, offset);
	printf("after cube, offset: %d\n", offset);
	printf("after cube, num: %d\n", num);

	#if 0
	if (num_old != nb_el) {
		printf("nb_el should equal num_old\n");
		exit(0);
	}

	if (num != num_old) {
		printf("Less than the full number of particles are used\n");
		printf("mismatch of num. Deal with it\n");
		printf("num, num_old= %d, %d\n", num, num_old);
		exit(1);
	}
	#endif

	num = offset;
	//printf("new num= %d\n", num); exit(0);

    cl_params = new BufferGE<GE_SPHParams>(ps->cli, 1);
	GE_SPHParams& params = *(cl_params->getHostPtr());
    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.rest_density = sph_settings.rest_density;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.particle_radius = sph_settings.particle_radius;
    params.simulation_scale = sph_settings.simulation_scale;
	printf("scale: %f\n", params.simulation_scale);

	// does scale_simulation influence stiffness and dampening?
    params.boundary_stiffness = 10000.;  //10000.0f;  (scale from 20000 to 20)
    params.boundary_dampening = 256.;//256.; 
    params.boundary_distance = sph_settings.boundary_distance;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 1.5f; //100.0f; //1.5f;
	params.dt = psfr->settings.dt;
	//printf("dt= %f\n", params.dt); exit(0);
 
	cl_params->copyToDevice();
	cl_params->copyToHost();
	GE_SPHParams& pparams = *(cl_params->getHostPtr());

//	printf("new num: %d\n", num); exit(0);

	// Decrease/increase num (nb particles as necessary)
    //*** Initialization, TODO: move out of here to the particle directory
	if (offset > num) {
		printf("offset should be <= num\n");
		exit(0);
	}
	num = offset;
	nb_el = num;
    std::vector<float4> colors(num);
    positions.resize(num);
    forces.resize(num);
    velocities.resize(num);
    densities.resize(num);

    std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.0f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);
    std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

	#if 0
	printf("h= %f\n", h);
    for(int i = 0; i < nb_el; i++)
    {
        printf("position[%d] = %f %f %f\n", positions[i].x, positions[i].y, positions[i].z);
    }
	exit(0);
    #endif

    //*** end Initialization

	// Put in setup Arrays
    // VBO creation, TODO: should be abstracted to another class
    managed = true;
    //printf("positions: %d, %d, %d\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    //printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    //printf("col vbo: %d\n", col_vbo);
    // end VBO creation

    //vbo buffers
    cl_position = new BufferVBO<float4>(ps->cli, pos_vbo);
    cl_color    = new BufferVBO<float4>(ps->cli, col_vbo);

    //pure opencl buffers
    cl_force       = new BufferGE<float4>(ps->cli, &forces[0], forces.size());
    cl_velocity    = new BufferGE<float4>(ps->cli, &velocities[0], velocities.size());
    cl_density     = new BufferGE<float>(ps->cli, &densities[0], densities.size());
    cl_error_check = new BufferGE<float4>(ps->cli, &error_check[0], error_check.size());

	setupArrays(); // From GE structures

	printf("=========================================\n");
	printf("Spacing =  particle_radius\n");
	printf("Smoothing distance = 2 * particle_radius to make sure we have particle-particle influence\n");
	printf("\n");
	printf("=========================================\n");
	params.print();
	sph_settings.print();
	cl_GridParams->getHostPtr()->print();
	cl_GridParamsScaled->getHostPtr()->print();
	cl_FluidParams->getHostPtr()->print();
	printf("=========================================\n");

	int print_freq = 20000;
	int time_offset = 5;

	ts_cl[TI_HASH]   = new GE::Time("hash",      time_offset, print_freq);
	ts_cl[TI_RADIX_SORT]   = new GE::Time("radix sort",   time_offset, print_freq);
	ts_cl[TI_BITONIC_SORT] = new GE::Time("bitonic sort", time_offset, print_freq);
	ts_cl[TI_BUILD]  = new GE::Time("build",     time_offset, print_freq);
	ts_cl[TI_NEIGH]  = new GE::Time("neigh",     time_offset, print_freq);
	ts_cl[TI_DENS]   = new GE::Time("density",   time_offset, print_freq);
	ts_cl[TI_PRES]   = new GE::Time("pressure",  time_offset, print_freq);
	ts_cl[TI_COL]      = new GE::Time("color",  time_offset, print_freq);
	ts_cl[TI_COL_NORM] = new GE::Time("color_norm", time_offset, print_freq);
	ts_cl[TI_VISC]   = new GE::Time("viscosity", time_offset, print_freq);
	ts_cl[TI_EULER]  = new GE::Time("euler",     time_offset, print_freq);
	ts_cl[TI_UPDATE] = new GE::Time("update",    time_offset, print_freq);
	ts_cl[TI_COLLISION_WALL] 
	                 = new GE::Time("collision wall",    time_offset, print_freq);
	//ps->setTimers(ts_cl);

	// copy pos, vel, dens into vars_unsorted()
	// COULD DO THIS ON GPU
	float4* vars = cl_vars_unsorted->getHostPtr();
	BufferGE<float4>& un = *cl_vars_unsorted;
	BufferGE<float4>& so = *cl_vars_sorted;

	for (int i=0; i < nb_el; i++) {
		//vars[i+DENS*num] = densities[i];
		// PROBLEM: density is float, but vars_unsorted is float4
		// HOW TO DEAL WITH THIS WITHOUT DOUBLING MEMORY ACCESS in 
		// buildDataStructures. 

		//printf("%d, %d, %d, %d\n", DENS, POS, VEL, FOR); exit(0);

		un(i+DENS*nb_el).x = densities[i];
		un(i+DENS*nb_el).y = 1.0; // for surface tension (always 1)
		un(i+DENS*nb_el).z = 0.0;
		un(i+DENS*nb_el).w = 0.0;
		un(i+POS*nb_el) = positions[i];
		un(i+VEL*nb_el) = velocities[i];
		un(i+FOR*nb_el) = forces[i];

		// SHOULD NOT BE REQUIRED
		so(i+DENS*nb_el).x = densities[i];
		so(i+DENS*nb_el).y = 1.0;  // for surface tension (always 1)
		so(i+DENS*nb_el).z = 0.0;
		so(i+DENS*nb_el).w = 0.0;
		so(i+POS*nb_el) = positions[i];
		so(i+VEL*nb_el) = velocities[i];
		so(i+FOR*nb_el) = forces[i];
	}

	cl_vars_unsorted->copyToDevice();
	cl_vars_sorted->copyToDevice(); // should not be required
}

//----------------------------------------------------------------------
GE_SPH::~GE_SPH()
{
	//printf("**** inside GE_SPH destructor ****\n");

    if(pos_vbo && managed)
    {
        glBindBuffer(1, pos_vbo);
        glDeleteBuffers(1, (GLuint*)&pos_vbo);
        pos_vbo = 0;
    }
    if(col_vbo && managed)
    {
        glBindBuffer(1, col_vbo);
        glDeleteBuffers(1, (GLuint*)&col_vbo);
        col_vbo = 0;
    }

	#if 1
	delete 	cl_vars_sorted;
	delete 	cl_vars_unsorted;
	delete 	cl_cells; // positions in Ian code
	delete 	cl_cell_indices_start;
	delete 	cl_cell_indices_end;
	delete 	cl_vars_sort_indices;
	delete 	cl_sort_hashes;
	delete 	cl_sort_indices;
	delete 	cl_unsort;
	delete 	cl_sort;
	delete  cl_GridParams;
	delete  cl_FluidParams;
	delete  cl_params;
	delete	clf_debug;  //just for debugging cl files
	delete	cli_debug;  //just for debugging cl files

    delete cl_velocity;
    delete cl_density;
	delete cl_position;
	delete cl_color;
	delete cl_force;
	#endif

	if (radixSort) delete radixSort;

	//printf("***** exit GE_SPH destructor **** \n");
}

//----------------------------------------------------------------------
void GE_SPH::update()
{
	static int count=0;

	ts_cl[TI_UPDATE]->start(); // OK

    //call kernels
    //TODO: add timings
#ifdef CPU
	computeOnCPU();
#endif

#ifdef GPU
	int nb_sub_iter = 15;
	computeOnGPU(nb_sub_iter);
	if (count % 10 == 0) computeTimeStep();
#endif

    /*
    std::vector<float4> ftest = cl_force->copyToHost(100);
    for(int i = 0; i < 100; i++)
    {
        if(ftest[i].z != 0.0)
            printf("force: %f %f %f  \n", ftest[i].x, ftest[i].y, ftest[i].z);
    }
    printf("execute!\n");
    */

	ts_cl[TI_UPDATE]->end(); // OK

	count++;
	//printf("count= %d\n", count);
	if (count%20 == 0) {
		//count = 0;
		printf("ITERATION: %d\n", count*nb_sub_iter);
		printf("count= %d, nb_sub_iter= %d\n", count, nb_sub_iter);
		GE::Time::printAll();
	}
}
//----------------------------------------------------------------------
void GE_SPH::setupArrays()
{
	printf("params: scale: %f\n", params.simulation_scale);
	GE_SPHParams& params = *(cl_params->getHostPtr());

	// only for my test routines: sort, hash, datastructures
	//printf("setupArrays, nb_el= %d\n", nb_el); exit(0);

	cl_cells = new BufferGE<float4>(ps->cli, nb_el);
	for (int i=0; i < nb_el; i++) {
		(*cl_cells)[i] = positions[i];
	}
	cl_cells->copyToDevice();

// Need an assign operator (no memory allocation)

	printf("allocate BufferGE<GridParams>\n");
	printf("sizeof(GridParams): %d\n", sizeof(GridParams));
	cl_GridParams = new BufferGE<GridParams>(ps->cli, 1); // destroys ...
	cl_GridParamsScaled = new BufferGE<GridParamsScaled>(ps->cli, 1); // destroys ...

	GridParams& gp = *(cl_GridParams->getHostPtr());
	printf("gp(host)= %ld\n", (long) &gp);

	GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());

	gp.grid_min = grid.getMin();
	gp.grid_max = grid.getMax();
	gp.grid_res = grid.getRes();
	gp.grid_size = grid.getSize();
	printf("*** grid_size= %d\n", grid_size);
	gp.grid_delta = grid.getDelta();
	gp.numParticles = nb_el;

	gp.grid_inv_delta.x = 1. / gp.grid_delta.x;
	gp.grid_inv_delta.y = 1. / gp.grid_delta.y;
	gp.grid_inv_delta.z = 1. / gp.grid_delta.z;
	gp.grid_inv_delta.w = 1.;
	gp.grid_inv_delta.print("inv delta");
	gp.nb_vars = nb_vars;

	cl_GridParams->copyToDevice();

	gp.grid_size.print("grid size (domain dimensions)"); // domain dimensions
	gp.grid_delta.print("grid delta (cell size)"); // cell size
	gp.grid_min.print("grid min");
	gp.grid_max.print("grid max");
	gp.grid_res.print("grid res (nb points)"); // number of points
	gp.grid_delta.print("grid delta");
	gp.grid_inv_delta.print("grid inv delta");

    float ss = params.simulation_scale;
	//ss = 1.0; // TEMPORARY UNTIL I KNOW WHAT I AM DOING
	printf("ss= %f\n", ss);
//	exit(0);

	gps.grid_size.x = gp.grid_size.x * ss;
	gps.grid_size.y = gp.grid_size.y * ss;
	gps.grid_size.z = gp.grid_size.z * ss;
	gps.grid_delta.x = gp.grid_delta.x * ss;
	gps.grid_delta.y = gp.grid_delta.y * ss;
	gps.grid_delta.z = gp.grid_delta.z * ss;
	gps.grid_min.x = gp.grid_min.x * ss;
	gps.grid_min.y = gp.grid_min.y * ss;
	gps.grid_min.z = gp.grid_min.z * ss;
	gps.grid_max.x = gp.grid_max.x * ss;
	gps.grid_max.y = gp.grid_max.y * ss;
	gps.grid_max.z = gp.grid_max.z * ss;
	gps.grid_res.x = gp.grid_res.x;
	gps.grid_res.y = gp.grid_res.y;
	gps.grid_res.z = gp.grid_res.z;
	gps.grid_inv_delta.x = gp.grid_inv_delta.x / ss;
	gps.grid_inv_delta.y = gp.grid_inv_delta.y / ss;
	gps.grid_inv_delta.z = gp.grid_inv_delta.z / ss;
	gps.grid_inv_delta.w = 1.0;
	gps.nb_vars = nb_vars;
	gps.numParticles = nb_el;

	gp.print();
	gps.print();
	//printf("gps.grid_size.x = %f\n", gps.grid_size.x);
	//exit(0);

	cl_GridParamsScaled->copyToDevice();

	cl_vars_unsorted = new BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_vars_sorted   = new BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_sort_indices  = new BufferGE<int>(ps->cli, nb_el);
	cl_sort_hashes   = new BufferGE<int>(ps->cli, nb_el);

	// ERROR
	int grid_size = nb_cells.x * nb_cells.y * nb_cells.z;
	printf("grid_size= %d\n", grid_size);
	grid_size = 10000;
	if (grid_size > 10000000) {
		printf("nb cells: %d, %d, %d\n", nb_cells.x, nb_cells.y, nb_cells.z);
		printf("grid_size too large (> 10000000)\n");
		exit(0);
	}

	// Size is the grid size. That is a problem since the number of
	// occupied cells could be much less than the number of grid elements. 
	cl_cell_indices_start = new BufferGE<int>(ps->cli, grid_size);
	cl_cell_indices_end   = new BufferGE<int>(ps->cli, grid_size);

	// For bitonic sort. Remove when bitonic sort no longer used
	// Currently, there is an error in the Radix Sort (just run both
	// sorts and compare outputs visually
	cl_sort_output_hashes = new BufferGE<int>(ps->cli, nb_el);
	cl_sort_output_indices = new BufferGE<int>(ps->cli, nb_el);


	clf_debug = new BufferGE<float4>(ps->cli, nb_el);
	cli_debug = new BufferGE<int4>(ps->cli, nb_el);


	int nb_floats = nb_vars*nb_el;
	// WHY NOT cl_float4 (in CL/cl_platform.h)
	float4 f;
	float4 zero;

	zero.x = zero.y = zero.z = 0.0;
	zero.w = 1.;


	// SETUP FLUID PARAMETERS
	// cell width is one diameter of particle, which imlies 27 neighbor searches
	#if 1
	cl_FluidParams = new BufferGE<FluidParams>(ps->cli, 1);
	FluidParams& fp = *cl_FluidParams->getHostPtr();;
	float radius = sph_settings.particle_radius;
	fp.smoothing_length = sph_settings.smoothing_distance; // SPH radius
	fp.scale_to_simulation = params.simulation_scale; // overall scaling factor
	//fp.mass = 1.0; // mass of single particle (MIGHT HAVE TO BE CHANGED)
	fp.friction_coef = 0.1;
	fp.restitution_coef = 0.9;
	fp.damping = 0.9;
	fp.shear = 0.9;
	fp.attraction = 0.9;
	fp.spring = 0.5;
	fp.gravity = -9.8; // -9.8 m/sec^2
	fp.choice = 0; // compute density
	cl_FluidParams->copyToDevice();
	#endif

    printf("done with setup arrays\n");
}
//----------------------------------------------------------------------
void GE_SPH::checkDensity()
{
#if 0
        //test density
		// Density checks should be in Density.cpp I believe (perhaps not)
        std::vector<float> dens = cl_density->copyToHost(num);
        float dens_sum = 0.0f;
        for(int j = 0; j < num; j++)
        {
            dens_sum += dens[j];
        }
        printf("summed density: %f\n", dens_sum);
        /*
        std::vector<float4> er = cl_error_check->copyToHost(10);
        for(int j = 0; j < 10; j++)
        {
            printf("rrrr[%d]: %f %f %f %f\n", j, er[j].x, er[j].y, er[j].z, er[j].w);
        }
        */
#endif
}
//----------------------------------------------------------------------
void GE_SPH::computeOnGPU(int nb_sub_iter)
{
    glFinish();
    cl_position->acquire();
    cl_color->acquire();
    
	nb_sub_iter = 1;
    for(int i=0; i < nb_sub_iter; i++)
    {
		//printf("i= %d\n", i);
		// ***** Create HASH ****
		hash();
		//exit(0);

		// **** Sort arrays ****
		// only power of 2 number of particles
		//radix_sort();
		bitonic_sort();

		// **** Reorder pos, vel
		buildDataStructures(); 

		#if 1
		// ***** DENSITY UPDATE *****
		neighborSearch(0); //density

		// ***** DENSITY DENOMINATOR *****
		//   *** DENSITY NORMALIZATION ***
		//neighborSearch(3); 

		// ***** COLOR GRADIENT *****
		neighborSearch(2); 

		// ***** PRESSURE UPDATE *****
		neighborSearch(1); //pressure
		#endif

		// ***** WALL COLLISIONS *****
		computeCollisionWall();

        // ***** EULER UPDATE *****
		computeEuler();

		// *** OUTPUT PHYSICAL VARIABLES FROM THE GPU
	//	exit(0);

	}

	//printGPUDiagnostics();
	//exit(0);

    cl_position->release();
    cl_color->release();
}
//----------------------------------------------------------------------
void GE_SPH::computeOnCPU()
{
    cpuDensity();
    //cpuPressure();
    //cpuEuler();
}
//----------------------------------------------------------------------
float GE_SPH::computeTimeStep()
{
	cl_vars_unsorted->copyToHost();
	float4* vel = cl_vars_unsorted->getHostPtr() + 2*nb_el;
	GridParams* gp = cl_GridParams->getHostPtr();
	GE_SPHParams* params = cl_params->getHostPtr();

	float velmax = 0.;
	float vel2;
	for (int i=0; i < nb_el; i++) {
		vel2 = vel[i].x*vel[i].x+vel[i].y*vel[i].y+vel[i].z*vel[i].z;
		velmax = vel2 > velmax ? vel2 : velmax;
	}
	velmax = sqrt(velmax);
	float soundSpeed = sqrt(params->K);
	float dt = 0.5 * gp->grid_delta.x / (velmax+soundSpeed);

	printf("velmax = %f\n", velmax);
	printf("time step limit: %f\n", dt);
	return dt;
}
//----------------------------------------------------------------------
void GE_SPH::printGPUDiagnostics()
{
		float4* pos;

		#if 0
		//Update position array (should be done on GPU)
		cl_vars_unsorted->copyToHost();
		pos  = cl_vars_unsorted->getHostPtr() + 1*nb_el;
		for (int i=0; i < nb_el; i++) {
			//printf("i= %d\n", i);
			positions[i] = pos[i];
		}
		#endif


		#if 1
		//  print out density
		cl_vars_unsorted->copyToHost();
		cl_vars_sorted->copyToHost();

		float4* density = cl_vars_unsorted->getHostPtr() + 0*nb_el;
		pos     = cl_vars_unsorted->getHostPtr() + 1*nb_el;
		float4* vel     = cl_vars_unsorted->getHostPtr() + 2*nb_el;
		float4* force   = cl_vars_unsorted->getHostPtr() + 3*nb_el;

		float4* density1 = cl_vars_sorted->getHostPtr() + 0*nb_el;
		float4* pos1     = cl_vars_sorted->getHostPtr() + 1*nb_el;
		float4* vel1     = cl_vars_sorted->getHostPtr() + 2*nb_el;
		float4* force1   = cl_vars_sorted->getHostPtr() + 3*nb_el;

		for (int i=1000; i < 1500; i++) {
		//for (int i=0; i < 10; i++) {
			printf("=== i= %d ==========\n", i);
			//printf("dens[%d]= %f, sorted den: %f\n", i, density[i].x, density1[i].x);
			pos[i].print("un pos");
			vel[i].print("un vel");
			force[i].print("un force");
			//density[i].print("un density");
			//pos1[i].print("so pos1");
			//vel1[i].print("so vel1");
			//force1[i].print("so force1");
		} 
		//exit(0);
		#endif
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

}
