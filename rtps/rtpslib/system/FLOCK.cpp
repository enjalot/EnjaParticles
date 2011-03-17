
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "FLOCK.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

//for random
#include<time.h>

namespace rtps {


FLOCK::FLOCK(RTPS *psfr, int n)
{
    //store the particle system framework
    ps = psfr;

    max_num = n;
    num = 0;
    nb_var = 10;

    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);
    veleval.resize(max_num);
    densities.resize(max_num);
    xflocks.resize(max_num);

    //seed random
    srand ( time(NULL) );

    grid = ps->settings.grid;

    //FLOCK settings depend on number of particles used
    calculateFLOCKSettings();
    //set up the grid
    setupDomain();

    //flock_settings.integrator = LEAPFROG2;
    flock_settings.integrator = EULER2;

    //*** end Initialization

    setupTimers();

    //setup the sorted and unsorted arrays
    prepareSorted();

#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");



    //replace these with the 2 steps
    /*
    loadDensity();
    loadPressure();
    loadViscosity();
    loadXFLOCK();
    */

    //loadStep1();
    //loadStep2();

    loadCollision_wall();
    loadCollision_tri();

    //could generalize this to other integration methods later (leap frog, RK4)
    if(flock_settings.integrator == LEAPFROG2)
    {
        loadLeapFrog();
    }
    else if(flock_settings.integrator == EULER2)
    {
        loadEuler();
    }

    loadScopy();

    loadPrep();
    loadHash();
    loadBitonicSort();
    loadDataStructures();
    loadNeighbors();

    ////////////////// Setup some initial particles
    //// really this should be setup by the user
    //int nn = 1024;
    int nn = 3333;
    //nn = 8192;
    nn = 2048;
    //nn = 1024;
    //float4 min = float4(.4, .4, .1, 0.0f);
    //float4 max = float4(.6, .6, .4, 0.0f);


    //float4 min   = float4(-559., -15., 0.5, 1.);
	//float4 max   = float4(-400., 225., 1050., 1);
    //grid = Domain(float4(-560,-30,0,0), float4(256, 256, 1276, 0));
    //float4 min   = float4(100./sd, -15./sd, 0.5/sd, 1.);
    //float4 min   = float4(100., -15., 550, 1.);
	//float4 max   = float4(255./sd, 225./sd, 1250./sd, 1);



    //float4 min = float4(.1, .1, .1, 0.0f);
    //float4 max = float4(.3, .3, .4, 0.0f);

    //addBox(nn, min, max, false);
    
    //nn = 512;
    /*nn = 512;
    float4 min = float4(.1, .1, .1, 1.0f);
	float4 max = float4(.9, .5, .9, 1.0f);
    addBox(nn, min, max, false);*/
    //addBox(nn, min, max, false);
   
    //float4 center = float4(.1/scale, .15/scale, .3/scale, 0.0f);
    //addBall(nn, center, .06/scale);
    //addBall(nn, center, .1/scale);
    ////////////////// Done with setup particles

    ////DEBUG STUFF
    printf("positions 0: \n");
    positions[0].print();
    
    //////////////


    
    
#endif
     // settings defaults to 0
     renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
     printf("spacing for radius %f\n", flock_settings.spacing);
     //renderer->setParticleRadius(spacing*0.5);
     //renderer->setParticleRadius(spacing*0.5);
     renderer->setParticleRadius(flock_settings.spacing);
	 
     //renderer = new Render(pos_vbo,col_vbo,num,ps->cli);
     //renderer->setParticleRadius(flock_settings.spacing*0.25);

}

FLOCK::~FLOCK()
{
    printf("FLOCK destructor\n");
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

}

void FLOCK::update()
{
    //call kernels
    //TODO: add timings
#ifdef CPU
    updateCPU();
#endif
#ifdef GPU
    updateGPU();
#endif
}

//----------------------------------------------------------------------
void FLOCK::updateCPU()
{
    //cpuDensity();
    //cpuPressure();
    //cpuViscosity();
    //cpuXFLOCK();
    //cpuCollision_wall();

    if(flock_settings.integrator == EULER2)
    {
        //cpuEuler();   // Original from Myrna
		// Modified by Gordon Erlebacher, 
        ge_cpuEuler();  // based on my boids program
    }
    else if(flock_settings.integrator == LEAPFROG2)
    {
        cpuLeapFrog();
    }
    //printf("positions[0].z %f\n", positions[0].z);
#if 0
    for(int i = 0; i < 10; i+=15)
    {
        //if(xflocks[i].z != 0.0)
        printf("particle %d, positions: %f %f %f  \n", positions[i].x, positions[i].y, positions[i].z);
        //printf("force: %f %f %f  \n", forces[i].x, forces[i].y, forces[i].z);
        //printf("force: %f %f %f  \n", velocities[i].x, velocities[i].y, velocities[i].z);
    }
#endif
    //printf("cpu execute!\n");

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
}

//----------------------------------------------------------------------
void FLOCK::updateGPU()
{

    timers[TI_UPDATE]->start();
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
    
    //sub-intervals
    int sub_intervals = 10;  //should be a setting
    for(int i=0; i < sub_intervals; i++)
    {
        /*
        k_density.execute(num);
        k_pressure.execute(num);
        k_viscosity.execute(num);
        k_xflock.execute(num);
        */
        //printf("hash\n");
        timers[TI_HASH]->start();
        hash();
        timers[TI_HASH]->end();
        //printf("bitonic_sort\n");
        timers[TI_BITONIC_SORT]->start();
        bitonic_sort();
        timers[TI_BITONIC_SORT]->end();
        //printf("data structures\n");
        timers[TI_BUILD]->start();
        buildDataStructures(); //reorder
        timers[TI_BUILD]->end();
        
        timers[TI_NEIGH]->start();
        //printf("density\n");
        timers[TI_DENS]->start();
        neighborSearch(0);  //density => flockmates
        timers[TI_DENS]->end();
        //printf("forces\n");
        //timers[TI_FORCE]->start();
        //neighborSearch(1);  //forces => velocities
        //timers[TI_FORCE]->end();
        //exit(0);
        timers[TI_NEIGH]->end();
        
        //printf("collision\n");
//        collision();
        //printf("integrate\n");
        integrate();		// compute the rules and itegrate
        //exit(0);
        //
        //Andrew's rendering emporium
        //neighborSearch(4);
    }

    cl_position.release();
    cl_color.release();

    timers[TI_UPDATE]->end();

}

void FLOCK::collision()
{
    int local_size = 128;
    //when implemented other collision routines can be chosen here
    timers[TI_COLLISION_WALL]->start();
    k_collision_wall.execute(num, local_size);
    timers[TI_COLLISION_WALL]->end();

    timers[TI_COLLISION_TRI]->start();
    collide_triangles();
    timers[TI_COLLISION_TRI]->end();

}

void FLOCK::integrate()
{
    int local_size = 128;
    if(flock_settings.integrator == EULER2)
    {
        //k_euler.execute(max_num);
        timers[TI_EULER]->start();
        k_euler.execute(num, local_size);
        timers[TI_EULER]->end();
    }
    else if(flock_settings.integrator == LEAPFROG2)
    {
        //k_leapfrog.execute(max_num);
        timers[TI_LEAPFROG]->start();
        k_leapfrog.execute(num, local_size);
        timers[TI_LEAPFROG]->end();
    }


#if 0
    if(num > 0)
    {
        std::vector<float4> pos = cl_position.copyToHost(num);
        //std::vector<float4> vel = cl_velocity.copyToHost(num);
        std::vector<int4> cli = cli_debug.copyToHost(num);
        std::vector<float4> clf = clf_debug.copyToHost(num);

        //std::vector<float4> f = cl_force.copyToHost(num);
        //std::vector<float4> d = cl_density.copyToHost(num);
        //std::vector<float4> xf = cl_xflock.copyToHost(num);
        for(int i = 0; i < num; i+=128)
        {
            printf("pos   [%d] = %f %f %f\n", i, pos[i].x, pos[i].y, pos[i].z);
            //printf("vel   [%d] = %f %f %f\n", i, vel[i].x, vel[i].y, vel[i].z);
            //printf("ne flo[%d] = %d %d \n", i, cli[i].x, cli[i].y);
            //printf("ve flo[%d] = %f %f %f\n", i, clf[i].w, clf[i].y, clf[i].z);
        }

    }
#endif



}

int FLOCK::setupTimers()
{
    //int print_freq = 20000;
    int print_freq = 1000; //one second
    int time_offset = 5;

    timers[TI_UPDATE]     = new GE::Time("update", time_offset, print_freq);
    timers[TI_HASH]     = new GE::Time("hash", time_offset, print_freq);
    timers[TI_BUILD]     = new GE::Time("build", time_offset, print_freq);
    timers[TI_BITONIC_SORT]     = new GE::Time("bitonic_sort", time_offset, print_freq);
    timers[TI_NEIGH]     = new GE::Time("neigh", time_offset, print_freq);
    timers[TI_DENS]     = new GE::Time("dens", time_offset, print_freq);
    timers[TI_FORCE]     = new GE::Time("force", time_offset, print_freq);
    timers[TI_COLLISION_WALL]     = new GE::Time("collision_wall", time_offset, print_freq);
    timers[TI_COLLISION_TRI]     = new GE::Time("collision_triangle", time_offset, print_freq);
    timers[TI_EULER]     = new GE::Time("euler", time_offset, print_freq);
    timers[TI_LEAPFROG]     = new GE::Time("leapfrog", time_offset, print_freq);
}

void FLOCK::printTimers()
{
    for(int i = 0; i < 11; i++) //switch to vector of timers and use size()
    {
        timers[i]->print();
    }
    System::printTimers();
}

void FLOCK::calculateFLOCKSettings()
{
    /*!
    * The Particle Mass (and hence everything following) depends on the MAXIMUM number of particles in the system
    */

    float4 dmin = grid.getBndMin();
    float4 dmax = grid.getBndMax();
    //printf("dmin: %f %f %f\n", dmin.x, dmin.y, dmin.z);
    //printf("dmax: %f %f %f\n", dmax.x, dmax.y, dmax.z);
    float domain_vol = (dmax.x - dmin.x) * (dmax.y - dmin.y) * (dmax.z - dmin.z);
    printf("domain volume: %f\n", domain_vol);


//////
    flock_settings.rest_density = 1000;
    //flock_settings.rest_density = 2000;

    flock_settings.particle_mass = (128*1024.0)/max_num * .0002;
    printf("particle mass: %f\n", flock_settings.particle_mass);

    float particle_vol = flock_settings.particle_mass / flock_settings.rest_density;
/////

    
    //constant .87 is magic
    //flock_settings.particle_rest_distance = .87 * pow(particle_vol, 1./3.);
    flock_settings.particle_rest_distance = .05;
    printf("particle rest distance: %f\n", flock_settings.particle_rest_distance);
    
    //messing with smoothing distance, making it really small to remove interaction still results in weird force values
    flock_settings.smoothing_distance = 2.0f * flock_settings.particle_rest_distance;

////
    flock_settings.boundary_distance = .5f * flock_settings.particle_rest_distance;
////







    //flock_settings.simulation_scale = pow(particle_vol * max_num / domain_vol, 1./3.); 
    flock_settings.simulation_scale = 1.0f;
    printf("simulation scale: %f\n", flock_settings.simulation_scale);

    flock_settings.spacing = flock_settings.particle_rest_distance/ flock_settings.simulation_scale;
    //flock_settings.spacing = .05;

    float particle_radius = flock_settings.spacing;
    printf("particle radius: %f\n", particle_radius);
 
    params.grid_min = dmin;
    params.grid_max = dmax;
    params.mass = flock_settings.particle_mass;
    params.rest_distance = flock_settings.particle_rest_distance;
    params.smoothing_distance = flock_settings.smoothing_distance;
    params.simulation_scale = flock_settings.simulation_scale;
    params.boundary_stiffness = 20000.0f;
    params.boundary_dampening = 256.0f;
    params.boundary_distance = flock_settings.particle_rest_distance * .5f;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 20.0f;
    params.num = num;
//    params.surface_threshold = 2.0 * params.simulation_scale; //0.01;
    params.viscosity = .001f;
    //params.viscosity = 1.0f;
    params.gravity = -9.8f;
    //params.gravity = 0.0f;
    params.velocity_limit = 600.0f;
    params.xflock_factor = .05f;

	float h = params.smoothing_distance;
	float pi = acos(-1.0);
	float h9 = pow(h,9.);
	float h6 = pow(h,6.);
	float h3 = pow(h,3.);
    params.wpoly6_coef = 315.f/64.0f/pi/h9;
	params.wpoly6_d_coef = -945.f/(32.0f*pi*h9);
	params.wpoly6_dd_coef = -945.f/(32.0f*pi*h9);
	params.wspiky_coef = 15.f/pi/h6;
    params.wspiky_d_coef = 45.f/(pi*h6);
	params.wvisc_coef = 15./(2.*pi*h3);
	params.wvisc_d_coef = 15./(2.*pi*h3);
	params.wvisc_dd_coef = 45./(pi*h6);

// debug mymese
float4 gmin = params.grid_min;
float4 gmax = params.grid_max;
float bd = params.boundary_distance;
printf("\n *************** \n boundary distance: %f\n", bd); 
printf("min grid: %f, %f, %f\n", gmin.x, gmin.y, gmin.z);
printf("max grid: %f, %f, %f\n ************** \n", gmax.x,gmax.y, gmax.z);

}



void FLOCK::prepareSorted()
{
    #include "flock/cl_macros.h"
    
    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);
 
    std::fill(forces.begin(), forces.end(),float4(0.0f, 1.0f, 0.0f, 0.0f));
    //std::fill(velocities.begin(), velocities.end(),float4(rand(), rand(), rand(), 0.0f));
    //std::fill(velocities.begin(), velocities.end(), float4(0.1f, 0.1f, 0.1f, 0.f));
    std::fill(velocities.begin(), velocities.end(), float4(1.f, 0.0f, 0.0f, 0.f));
    std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);
    std::fill(xflocks.begin(), xflocks.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

    // VBO creation, TODO: should be abstracted to another class
    managed = true;
    printf("positions: %zd, %zd, %zd\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);
    // end VBO creation

#ifdef GPU
    //vbo buffers
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);

    //pure opencl buffers: these are deprecated
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);
    cl_veleval = Buffer<float4>(ps->cli, veleval);
    cl_density = Buffer<float>(ps->cli, densities);
    cl_xflock = Buffer<float4>(ps->cli, xflocks);

    //cl_error_check= Buffer<float4>(ps->cli, error_check);

    //TODO make a helper constructor for buffer to make a cl_mem from a struct
    std::vector<FLOCKParams> vparams(0);
    vparams.push_back(params);
    cl_FLOCKParams = Buffer<FLOCKParams>(ps->cli, vparams);

    //Setup Grid Parameter structs
    std::vector<GridParams> gparams(0);
    gparams.push_back(grid_params);
    cl_GridParams = Buffer<GridParams>(ps->cli, gparams);
    //scaled Grid Parameters
    std::vector<GridParams> sgparams(0);
    sgparams.push_back(grid_params_scaled);
    cl_GridParamsScaled = Buffer<GridParams>(ps->cli, sgparams);

    //setup debug arrays
    std::vector<float4> clfv(max_num);
    std::fill(clfv.begin(), clfv.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<int4> cliv(max_num);
    std::fill(cliv.begin(), cliv.end(),int4(0.0f, 0.0f, 0.0f, 0.0f));
    clf_debug = Buffer<float4>(ps->cli, clfv);
    cli_debug = Buffer<int4>(ps->cli, cliv);

    //sorted and unsorted arrays
    std::vector<float4> unsorted(max_num*nb_var);
    std::vector<float4> sorted(max_num*nb_var);

    std::fill(unsorted.begin(), unsorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
    std::fill(sorted.begin(), sorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));

    cl_vars_unsorted = Buffer<float4>(ps->cli, unsorted);
    cl_vars_sorted = Buffer<float4>(ps->cli, sorted);

    std::vector<int> keys(max_num);
    //to get around limits of bitonic sort only handling powers of 2
#include "limits.h"
    std::fill(keys.begin(), keys.end(), INT_MAX);
	cl_sort_indices  = Buffer<int>(ps->cli, keys);
	cl_sort_hashes   = Buffer<int>(ps->cli, keys);

	// for debugging. Store neighbors of indices
	// change nb of neighbors in cl_macro.h as well
	//cl_index_neigh = Buffer<int>(ps->cli, max_num*50);

    // Size is the grid size. That is a problem since the number of
	// occupied cells could be much less than the number of grid elements. 
    std::vector<int> gcells(grid_params.nb_cells);
	int minus = 0xffffffff;
    std::fill(gcells.begin(), gcells.end(), 0);

	cl_cell_indices_start = Buffer<int>(ps->cli, gcells);
	cl_cell_indices_end   = Buffer<int>(ps->cli, gcells);
	//printf("gp.nb_points= %d\n", gp.nb_points); exit(0);

	// For bitonic sort. Remove when bitonic sort no longer used
	// Currently, there is an error in the Radix Sort (just run both
	// sorts and compare outputs visually
	cl_sort_output_hashes = Buffer<int>(ps->cli, keys);
	cl_sort_output_indices = Buffer<int>(ps->cli, keys);

    std::vector<Triangle> maxtri(2048);
    cl_triangles = Buffer<Triangle>(ps->cli, maxtri);
#endif

}

void FLOCK::setupDomain()
{
    grid.calculateCells(flock_settings.smoothing_distance / flock_settings.simulation_scale);


	grid_params.grid_min = grid.getMin();
	grid_params.grid_max = grid.getMax();
	grid_params.bnd_min  = grid.getBndMin();
	grid_params.bnd_max  = grid.getBndMax();
	grid_params.grid_res = grid.getRes();
	grid_params.grid_size = grid.getSize();
	grid_params.grid_delta = grid.getDelta();
	grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

    printf("gp nb_cells: %d\n", grid_params.nb_cells);


// debug mymese
float4 gmin = grid_params.bnd_min;
float4 gmax = grid_params.bnd_max;
float4 bd = grid_params.grid_size;
printf("\n *************** \n grid size: %f, %f, %f\n", bd.x, bd.y, bd.z); 
printf("min boundary: %f, %f, %f\n", gmin.x, gmin.y, gmin.z);
printf("max boundary: %f, %f, %f\n ************** \n", gmax.x,gmax.y, gmax.z);


    /*
	grid_params.grid_inv_delta.x = 1. / grid_params.grid_delta.x;
	grid_params.grid_inv_delta.y = 1. / grid_params.grid_delta.y;
	grid_params.grid_inv_delta.z = 1. / grid_params.grid_delta.z;
	grid_params.grid_inv_delta.w = 1.;
    */

    float ss = flock_settings.simulation_scale;

	grid_params_scaled.grid_min = grid_params.grid_min * ss;
	grid_params_scaled.grid_max = grid_params.grid_max * ss;
	grid_params_scaled.bnd_min  = grid_params.bnd_min * ss;
	grid_params_scaled.bnd_max  = grid_params.bnd_max * ss;
	grid_params_scaled.grid_res = grid_params.grid_res;
	grid_params_scaled.grid_size = grid_params.grid_size * ss;
	grid_params_scaled.grid_delta = grid_params.grid_delta / ss;
	//grid_params_scaled.nb_cells = (int) (grid_params_scaled.grid_res.x*grid_params_scaled.grid_res.y*grid_params_scaled.grid_res.z);
    grid_params_scaled.nb_cells = grid_params.nb_cells;
    //grid_params_scaled.grid_inv_delta = grid_params.grid_inv_delta / ss;
    //grid_params_scaled.grid_inv_delta.w = 1.0f;

    grid_params.print();
    grid_params_scaled.print();
    
}

int FLOCK::addBox(int nn, float4 min, float4 max, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_settings.simulation_scale;
    }
printf("\n\n ADDING A CUBE \n\n");
    vector<float4> rect = addRect(nn, min, max, flock_settings.spacing, scale);
    pushParticles(rect);
    return rect.size();
}

void FLOCK::addBall(int nn, float4 center, float radius, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_settings.simulation_scale;
    }
printf("\n\n ADDING A SPHERE \n\n");
    vector<float4> flockere = addSphere(nn, center, radius, flock_settings.spacing, scale);
    pushParticles(flockere);
}

void FLOCK::pushParticles(vector<float4> pos)
{
    int nn = pos.size();
    if (num + nn > max_num) {return;}
    float rr = (rand() % 255)/255.0f;
    float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
    
    //printf("random color: %f %f %f\n", rr, 0.f, 1.0f-rr, 1.f);
	//float4 color(0.0f,0.0f,0.1f,0.1f);

    std::vector<float4> cols(nn);
    std::vector<float4> vels(nn);

    std::fill(cols.begin(), cols.end(),color);
    //float v = .5f;
    float v = rand()/RAND_MAX;
    //float4 iv = float4(v, v, -v, 0.0f);
    float4 iv = float4(v, 0, -v, 0.0f);
    std::fill(vels.begin(), vels.end(),iv);

#ifdef CPU
 std::copy(pos.begin(), pos.end(), positions.begin()+num);
//printf("pushParticles nn = %d\n", nn);
//exit(0);
#endif

#ifdef GPU
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
 
    //printf("about to prep 0\n");
    prep(0);
    //printf("done with prep 0\n");

    cl_position.copyToDevice(pos, num);
    cl_color.copyToDevice(cols, num);

    cl_color.release();
    cl_position.release();

    //2 is from cl_macros.h should probably not hardcode this number
    cl_velocity.copyToDevice(vels, num);
    //cl_vars_unsorted.copyToDevice(vels, max_num*8+num);

    //params.num = num+nn;
    //std::vector<FLOCKParams> vparams(0);
    //vparams.push_back(params);
    //cl_FLOCKParams.copyToDevice(vparams);

    //printf("about to updateNum\n");
    //ps->updateNum(params.num);

    params.num = num+nn;
    updateFLOCKP();

    num += nn;  //keep track of number of particles we use

    cl_position.acquire();
    //reprep the unsorted (packed) array to account for new particles
    //might need to do it conditionally if particles are added or subtracted
    printf("about to prep\n");
    prep(1);
    printf("done with prep\n");
    cl_position.release();
#else
    //glFinish();
    //params.num = num+nn;
    //printf("about to updateNum CPU\n");
    //ps->updateNum(params.num);
    num += nn;  //keep track of number of particles we use
#endif
	renderer->setNum(num);
}
void FLOCK::updateFLOCKP()
{
    std::vector<FLOCKParams> vparams(0);
    vparams.push_back(params);
    cl_FLOCKParams.copyToDevice(vparams);
}

void FLOCK::render()
{
	System::render();
	renderer->render_box(grid.getBndMin(), grid.getBndMax());
    //renderer->render_table(grid.getBndMin(), grid.getBndMax());
}



} //end namespace
