
//----------------------------------------------------------------------
// code does not work
void GE_SPH::gordon_parameters()
{
	// STRATEGY
	// mass of single particle based on density and particle radius
	// mass of single cell of the grid, assuming np particles per cell
	//   (np is 1 or 8 in 3D)

	// Density (mass per unit vol)
	// density of water: 1000 kg/m^3  (62lb/ft^3)
	double density; // water: 62 lb/ft^3 = 1000 kg/m^3
	double particle_radius;
	double particle_volume;

	density = 1000.; 
	double pi = acos(-1.);
	double nb_particles = num;
	double rat = (float) (num/4096.);
	double particle_mass = particle_volume * density;
	particle_mass = 0.00020543 / rat; // from Fluids2
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
	float particle_rest_distance = 1.00 * 0.87*particle_size; // why 0.87? 
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
	// (4/3*pi*h^3)/ particle_spacing^3 = nb_interact_part
	double coef = 3.14159*4./3.;
	//double h = pow(nb_interact_part/coef, 1./3.) * particle_radius;
	double h = pow(nb_interact_part/coef, 1./3.) * particle_spacing;
	double hvol = (4.*pi/3.)*pow(h,3.);
	double particle_cell_vol = pow(particle_spacing,3.);
	printf("particle_spacing= %f\n", particle_spacing);
	printf("h= %f\n", h);
	printf("nb_part= %d\n", (int) (4.*pi/3. * pow(h/particle_spacing,3.)));
	printf("nb_part= %d\n", (int) (hvol/particle_cell_vol));
	//exit(0);

	//h = .02; // larger than I would want it, but same as Fluid v.2
	//h = .01; // larger than I would want it, but same as Fluid v.2
	// domain cell size
	float cell_sz;
	cell_sz = h;  // 27 neighbor search (only neighbors strictly necessary)
	printf("cell_size, delta_x= %f\n", cell_sz);
	printf("smoothing_length h= %f\n", h);

	//-------------------------------
	//SETUP GRID

    sph_settings.simulation_scale = 0.010; //0.004;

	double cell_size = cell_sz;
	printf("cell_size= %f\n", cell_size); 

	//cell_size = h;
	int    nb_cells_x; // number of cells along x
	int    nb_cells_y; 
	int    nb_cells_z;

	#if 0
	// Dam (repeat case from Fluids v2
	// Size in world space
	float4 domain_min = float4(-10., -5., 0., 1.);
	float4 domain_max = float4(+10., +5., 15., 1.);
	float4 fluid_min   = float4( 4.5, -4.8,  0.03, 1.);
	float4 fluid_max   = float4( 9.9, +4.8,  12., 1.);
	#endif

	#if 1
	// box of fluid at rest
	float4 domain_min = float4(4.5, -5., 0., 1.);
	float4 domain_max = float4(+10., +5., 25., 1.);
	float4 fluid_min   = float4( 4.5, -4.9,  0.03, 1.);
	float4 fluid_max   = float4( 9.9, +4.9,  25., 1.);
	#endif


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

	//---------------------------------------------------------
	// STRUCTURES NEEDED BY MY PROGRAM: 
	// GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());
	// cl_FluidParams = new BufferGE<FluidParams>(ps->cli, 1);
    // cl_params = new BufferGE<GE_SPHParams>(ps->cli, 1);

	// kern.setArg(iarg++, cl_GridParamsScaled->getDevicePtr());
	// kern.setArg(iarg++, cl_FluidParams->getDevicePtr());
	// kern.setArg(iarg++, cl_params->getDevicePtr());
	// 
	//---------------------------------------------------------
    //init sph stuff
    sph_settings.rest_density = density;
    sph_settings.particle_mass = particle_mass;

	// Do not know why required  REST DISTANCE
    sph_settings.particle_rest_distance = particle_rest_distance; 
    sph_settings.particle_spacing = particle_spacing;  // distance between particles
    sph_settings.smoothing_distance = h;   // CHECK THIS. Width of W function
    sph_settings.particle_radius = particle_radius; 

	// factor of 2 is TEMPORARY (should be 1)
    sph_settings.boundary_distance =  2.0 * sph_settings.particle_spacing / 2.;
    sph_settings.boundary_distance =  particle_radius;

	printf("domain size: %f, %f, %f\n", domain_size_x, domain_size_y, domain_size_z);


	//-------------------------------
    
    //init sph stuff
    sph_settings.rest_density = density;
    sph_settings.particle_mass = particle_mass;

	// Do not know why required  REST DISTANCE
    sph_settings.particle_rest_distance = particle_rest_distance; 
    sph_settings.particle_spacing = particle_spacing;  // distance between particles
    sph_settings.smoothing_distance = h;   // CHECK THIS. Width of W function
    sph_settings.particle_radius = particle_radius; 

	// factor of 2 is TEMPORARY (should be 1)
    sph_settings.boundary_distance =  2.0 * sph_settings.particle_spacing / 2.;
    sph_settings.boundary_distance =  particle_radius;

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

	grid.print();
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
	printf("world spacing= %f\n", spacing);

	grid.makeCube(&positions[0], pmin, pmax, spacing, num, offset);
	printf("after cube, offset: %d\n", offset);
	printf("after cube, num: %d\n", num);

	printf("2 nb_el= %d\n", nb_el);

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

	printf("36 nb_el= %d\n", nb_el);

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

	printf("37 nb_el= %d\n", nb_el);

	// does scale_simulation influence stiffness and dampening?
    params.boundary_stiffness = 10000.;  //10000.0f;  (scale from 20000 to 20)
    params.boundary_dampening = 1256.;//256.; 
    params.boundary_distance = sph_settings.boundary_distance;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 1.5f; //100.0f; //1.5f;
	params.dt = ps->settings.dt;
	//printf("dt= %f\n", params.dt); exit(0);
 
	//cl_params->copyToDevice();
	//cl_params->copyToHost();

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
    //std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

	#if 0
	printf("h= %f\n", h);
    for(int i = 0; i < nb_el; i++)
    {
        printf("position[%d] = %f %f %f\n", positions[i].x, positions[i].y, positions[i].z);
    }
	exit(0);
    #endif

	printf("3 nb_el= %d\n", nb_el);

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
    //cl_error_check = new BufferGE<float4>(ps->cli, &error_check[0], error_check.size());
}
//----------------------------------------------------------------------





























//----------------------------------------------------------------------
// Ian's code does work (in branch rtps)
void GE_SPH::ian_parameters() 
{
	// Density (mass per unit vol)
	// density of water: 1000 kg/m^3  (62lb/ft^3)
	double pi = acos(-1.);

	double nb_particles = num;
	double rest_density = 1000.; // water: 62 lb/ft^3 = 1000 kg/m^3
	double particle_volume;
	double simulation_scale = 0.001;
	printf("ian: num= %d\n", nb_particles);
	double particle_mass = (128*1024.0)/nb_particles * .0002;
	double particle_rest_distance = 0.87*pow(particle_mass/rest_density, 1./3.); 
	double smoothing_distance = 2.0*particle_rest_distance;
	double h = smoothing_distance;
	double boundary_distance = 0.5*particle_rest_distance;
	// world coordinates
	double particle_spacing_w = particle_rest_distance / simulation_scale;
	double particle_radius_w = particle_spacing_w;

	//--------------------------------


    cl_params = new BufferGE<GE_SPHParams>(ps->cli, 1);
	GE_SPHParams& params = *(cl_params->getHostPtr());

	params.rest_distance = particle_rest_distance;
	params.rest_density = rest_density;
	params.smoothing_distance = h;
	params.particle_radius = particle_radius_w;
    params.boundary_stiffness = 10000.;  
    params.boundary_dampening = 256.;
    params.simulation_scale = simulation_scale;
	params.boundary_distance = boundary_distance;
    params.EPSILON = .00001f;
	params.mass = particle_mass;
    params.PI = acos(-1.);
    params.dt = ps->settings.dt;
    params.K = 20;

printf("****** REST 1 *****\n");
params.print();
	

	//-------------------------------
	//SETUP GRID

    positions.resize(num);

	//float cell_sz;
	//cell_sz = h;  // 27 neighbor search (only neighbors strictly necessary)

	double cell_size = h;
	double cell_size_w = h / simulation_scale;
	printf("cell_size_w= %f, cell_size= %f\n", cell_size_w, cell_size); 
	//exit(0);

	//cell_size = h;
	int    nb_cells_x; // number of cells along x
	int    nb_cells_y; 
	int    nb_cells_z;

	#if 1
	// Dam (repeat case from Fluids v2
	// Size in world space

	//float4 domain_min  = float4(-500, 0, 0, 1);
	//float4 domain_max  = float4(256, 256, 512, 1);
	float4 domain_min  = float4(-264, -30, 0, 1);
	float4 domain_max  = float4(256, 286, 512, 1);
	//float4 domain_max  = float4(256, 256, 1276, 1);

	// displace by 1/2 particle spacing in world coordinates
	float4 fluid_min   = float4(0., 30., 30., 1.);
	float4 fluid_max   = float4(220., 220., 450., 1);
	#endif

	#if 0
	// box of fluid at rest
	// domain minimimum is extended inside UniformGrid
	// domain_min become wall boundaries
	float4 domain_min  = float4(0, 0, 0, 1);
	float4 domain_max  = float4(256, 256, 512, 1);

	// displace by 1/2 particle spacing in world coordinates
	float4 fluid_min   = float4(0., 0., 0., 1.);
	float4 fluid_max   = float4(256., 256., 512., 1);
	#endif

	double domain_size_x = domain_max.x - domain_min.x; 
	double domain_size_y = domain_max.y - domain_min.y; 
	double domain_size_z = domain_max.z - domain_min.z; 

    sph_settings.simulation_scale = simulation_scale;
	float world_cell_size = cell_size / sph_settings.simulation_scale;
	printf("world_cell_size= %f\n", world_cell_size);


	//.......

	nb_cells = int4(nb_cells_x, nb_cells_y, nb_cells_z, 1);
	int num_old = num;
    grid = UniformGrid(domain_min, domain_max, cell_size_w);
    //grid = UniformGrid(domain_min, domain_max, nb_cells, sph_settings.simulation_scale); 

	int offset = 0;
	grid.makeCube(&positions[0], fluid_min, fluid_max, particle_spacing_w, num, offset);

	grid.res.print("grid res");
	printf("offset= %d\n", offset);
	printf("cell_size_w= %f\n", cell_size_w);
	printf("particle_spacing_w= %f\n", particle_spacing_w);
	grid.print();
	//exit(0);

	//END SETUP GRID

	params.grid_min = domain_min;
	params.grid_max = domain_max;
	//-------------------------------
	//-------------------------------

printf("****** REST 2 *****\n");
params.print();





#if 0
    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.rest_density = sph_settings.rest_density;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.particle_radius = sph_settings.particle_radius;
    params.simulation_scale = sph_settings.simulation_scale;
	printf("scale: %f\n", params.simulation_scale);

	printf("37 nb_el= %d\n", nb_el);

	// does scale_simulation influence stiffness and dampening?
    params.boundary_stiffness = 10000.;  //10000.0f;  (scale from 20000 to 20)
    params.boundary_dampening = 1256.;//256.; 
    params.boundary_distance = sph_settings.boundary_distance;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 1.5f; //100.0f; //1.5f;
	params.dt = ps->settings.dt;
	//printf("dt= %f\n", params.dt); exit(0);
 
#endif



	double rat = (float) (num/4096.);
	particle_mass = particle_volume * rest_density;
	particle_mass = 0.00020543 / rat; // from Fluids2
	particle_volume = particle_mass / rest_density;
	double particle_radius = pow(particle_volume*3./(4.*pi), 1./3.);
	printf("particle radius= %f\n", particle_radius);

	int nb_particles_in_cell = 1;
	// mass of fluid in a single cell
	float cell_mass = nb_particles_in_cell*particle_mass;
	float cell_volume = cell_mass / rest_density; // = particle_volume

	// Cell contains nb_particles_in_cell of fluid
	// size of single fluid element viewed as a cube

	// particle "cube" with same mass as spherical particle
	float particle_size = pow(cell_volume, 1./3.);

	float particle_spacing;
	// particle_spacing of particles at t=0
	// rest distance between particles
	particle_spacing = particle_rest_distance; 

	printf("particle_spacing= %f\n", particle_spacing);
	printf("particle_rest_distance= %f\n", particle_rest_distance);

	// desired number of particles within the smoothing_distance sphere
	int nb_interact_part = 50;
	// (4/3*pi*h^3)/ particle_spacing^3 = nb_interact_part
	double coef = 3.14159*4./3.;
	//double h = pow(nb_interact_part/coef, 1./3.) * particle_radius;
	h = pow(nb_interact_part/coef, 1./3.) * particle_spacing;
	double hvol = (4.*pi/3.)*pow(h,3.);
	double particle_cell_vol = pow(particle_spacing,3.);
	printf("particle_spacing= %f\n", particle_spacing);
	printf("h= %f\n", h);
	printf("nb_part= %d\n", (int) (4.*pi/3. * pow(h/particle_spacing,3.)));
	printf("nb_part= %d\n", (int) (hvol/particle_cell_vol));
	//exit(0);

	//h = .02; // larger than I would want it, but same as Fluid v.2
	//h = .01; // larger than I would want it, but same as Fluid v.2
	// domain cell size
	//float cell_sz;
	//cell_sz = h;  // 27 neighbor search (only neighbors strictly necessary)
	//printf("cell_size, delta_x= %f\n", cell_sz);
	//printf("smoothing_length h= %f\n", h);

	//---------------------------------------------------------
	// STRUCTURES NEEDED BY MY PROGRAM: 
	// GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());
	// cl_FluidParams = new BufferGE<FluidParams>(ps->cli, 1);
    // cl_params = new BufferGE<GE_SPHParams>(ps->cli, 1);

	// kern.setArg(iarg++, cl_GridParamsScaled->getDevicePtr());
	// kern.setArg(iarg++, cl_FluidParams->getDevicePtr());
	// kern.setArg(iarg++, cl_params->getDevicePtr());
	// 
	//---------------------------------------------------------
    //init sph stuff
    sph_settings.rest_density = rest_density;
    sph_settings.particle_mass = particle_mass;

	// Do not know why required  REST DISTANCE
    sph_settings.particle_rest_distance = particle_rest_distance; 
    sph_settings.particle_spacing = particle_spacing;  // distance between particles
    sph_settings.smoothing_distance = h;   // CHECK THIS. Width of W function
    sph_settings.particle_radius = particle_radius; 

	// factor of 2 is TEMPORARY (should be 1)
    sph_settings.boundary_distance =  2.0 * sph_settings.particle_spacing / 2.;
    sph_settings.boundary_distance =  particle_radius;

	printf("domain size: %f, %f, %f\n", domain_size_x, domain_size_y, domain_size_z);


	//-------------------------------
    
    //init sph stuff
    sph_settings.rest_density = rest_density;
    sph_settings.particle_mass = particle_mass;

	// Do not know why required  REST DISTANCE
    sph_settings.particle_rest_distance = particle_rest_distance; 
    sph_settings.particle_spacing = particle_spacing;  // distance between particles
    sph_settings.smoothing_distance = h;   // CHECK THIS. Width of W function
    sph_settings.particle_radius = particle_radius; 

	// factor of 2 is TEMPORARY (should be 1)
    sph_settings.boundary_distance =  2.0 * sph_settings.particle_spacing / 2.;
    sph_settings.boundary_distance =  particle_radius;

	printf("domain size: %f, %f, %f\n", domain_size_x, domain_size_y, domain_size_z);
	//exit(0);

//---------------------------------------

printf("num= %d\n", num);

    //*** Initialization, TODO: move out of here to the particle directory
    positions.resize(num);
    forces.resize(num);
    velocities.resize(num);
    densities.resize(num);

	float4 pmin = fluid_min;
	float4 pmax = fluid_max;
	pmin.print("pmin");
	pmax.print("pmax");
	printf("particle_spacing: %f\n", sph_settings.particle_spacing);

	// INITIATE PARTICLE POSITIONS
	float spacing = sph_settings.particle_spacing/sph_settings.simulation_scale;
	printf("world spacing= %f\n", spacing);

//	grid.makeCube(&positions[0], pmin, pmax, spacing, num, offset);
	printf("after cube, offset: %d\n", offset);
	printf("after cube, num: %d\n", num);

	printf("2 nb_el= %d\n", nb_el);

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

	printf("36 nb_el= %d\n", nb_el);

	num = offset;
	//printf("new num= %d\n", num); exit(0);

	//cl_params->copyToDevice();
	//cl_params->copyToHost();

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
    //std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

	#if 0
	printf("h= %f\n", h);
    for(int i = 0; i < nb_el; i++)
    {
        printf("position[%d] = %f %f %f\n", positions[i].x, positions[i].y, positions[i].z);
    }
	exit(0);
    #endif

	printf("3 nb_el= %d\n", nb_el);

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
    //cl_error_check = new BufferGE<float4>(ps->cli, &error_check[0], error_check.size());
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
