
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "SPH.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

//for random
#include<time.h>

namespace rtps {


SPH::SPH(RTPS *psfr, int n)
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
    xsphs.resize(max_num);

    //seed random
    srand ( time(NULL) );

    grid = ps->settings.grid;

    //SPH settings depend on number of particles used
    calculateSPHSettings();
    //set up the grid
    setupDomain();

    sph_settings.integrator = LEAPFROG;
    //sph_settings.integrator = EULER;

    //*** end Initialization

    setupTimers();

#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");

    //setup the sorted and unsorted arrays
    prepareSorted();

    //replace these with the 2 steps
    /*
    loadDensity();
    loadPressure();
    loadViscosity();
    loadXSPH();
    */

    //loadStep1();
    //loadStep2();

    loadCollision_wall();
    loadCollision_tri();

    //could generalize this to other integration methods later (leap frog, RK4)
    if(sph_settings.integrator == LEAPFROG)
    {
        loadLeapFrog();
    }
    else if(sph_settings.integrator == EULER)
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

	renderer = new Render(pos_vbo,col_vbo,num,ps->cli);
        renderer->setParticleRadius(sph_settings.spacing*0.5);

}

SPH::~SPH()
{
    printf("SPH destructor\n");
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

void SPH::update()
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

void SPH::updateCPU()
{
    cpuDensity();
    cpuPressure();
    cpuViscosity();
    cpuXSPH();
    cpuCollision_wall();

    if(sph_settings.integrator == EULER)
    {
        cpuEuler();
    }
    else if(sph_settings.integrator == LEAPFROG)
    {
        cpuLeapFrog();
    }
    //printf("positions[0].z %f\n", positions[0].z);
    /*
    for(int i = 0; i < 100; i++)
    {
        //if(xsphs[i].z != 0.0)
        //printf("force: %f %f %f  \n", veleval[i].x, veleval[i].y, veleval[i].z);
        printf("force: %f %f %f  \n", xsphs[i].x, xsphs[i].y, xsphs[i].z);
        //printf("force: %f %f %f  \n", velocities[i].x, velocities[i].y, velocities[i].z);
    }
    */
    //printf("cpu execute!\n");

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);


}

void SPH::updateGPU()
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
        k_xsph.execute(num);
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
        neighborSearch(0);  //density
        timers[TI_DENS]->end();
        //printf("forces\n");
        timers[TI_FORCE]->start();
        neighborSearch(1);  //forces
        timers[TI_FORCE]->end();
        //exit(0);
        timers[TI_NEIGH]->end();
        
        //printf("collision\n");
        collision();
        //printf("integrate\n");
        integrate();
        //exit(0);
        //
        //Andrew's rendering emporium
        //neighborSearch(4);
    }

    cl_position.release();
    cl_color.release();

    timers[TI_UPDATE]->end();

}

void SPH::collision()
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

void SPH::integrate()
{
    int local_size = 128;
    if(sph_settings.integrator == EULER)
    {
        //k_euler.execute(max_num);
        timers[TI_EULER]->start();
        k_euler.execute(num, local_size);
        timers[TI_EULER]->end();
    }
    else if(sph_settings.integrator == LEAPFROG)
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
        for(int i = 0; i < num; i++)
        {
            printf("pos[%d] = %f %f %f\n", i, pos[i].x, pos[i].y, pos[i].z);
        }
    }
#endif


}

int SPH::setupTimers()
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

void SPH::printTimers()
{
    for(int i = 0; i < 11; i++) //switch to vector of timers and use size()
    {
        timers[i]->print();
    }
    System::printTimers();
}

void SPH::calculateSPHSettings()
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

    sph_settings.rest_density = 1000;
    //sph_settings.rest_density = 2000;

    sph_settings.particle_mass = (128*1024.0)/max_num * .0002;
    printf("particle mass: %f\n", sph_settings.particle_mass);

    float particle_vol = sph_settings.particle_mass / sph_settings.rest_density;

    //constant .87 is magic
    sph_settings.particle_rest_distance = .87 * pow(particle_vol, 1./3.);
    printf("particle rest distance: %f\n", sph_settings.particle_rest_distance);
    
    //messing with smoothing distance, making it really small to remove interaction still results in weird force values
    sph_settings.smoothing_distance = 2.0f * sph_settings.particle_rest_distance;
    sph_settings.boundary_distance = .5f * sph_settings.particle_rest_distance;
    printf("particle smoothing distance: %f\n", sph_settings.smoothing_distance);

    sph_settings.simulation_scale = pow(particle_vol * max_num / domain_vol, 1./3.); 
    printf("simulation scale: %f\n", sph_settings.simulation_scale);

    sph_settings.spacing = sph_settings.particle_rest_distance/ sph_settings.simulation_scale;
    printf("spacing: %f\n", sph_settings.spacing);

    float particle_radius = sph_settings.spacing;
    printf("particle radius: %f\n", particle_radius);
 
    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.simulation_scale = sph_settings.simulation_scale;
    params.boundary_stiffness = 20000.0f;
    params.boundary_dampening = 256.0f;
    params.boundary_distance = sph_settings.particle_rest_distance * .5f;
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
    params.xsph_factor = .05f;

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

}

void SPH::prepareSorted()
{
    #include "sph/cl_macros.h"
    
    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);
 
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);
    std::fill(xsphs.begin(), xsphs.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

    // VBO creation, TODO: should be abstracted to another class
    managed = true;
    printf("positions: %zd, %zd, %zd\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);
    // end VBO creation

    //vbo buffers
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);

    //pure opencl buffers: these are deprecated
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);
    cl_veleval = Buffer<float4>(ps->cli, veleval);
    cl_density = Buffer<float>(ps->cli, densities);
    cl_xsph = Buffer<float4>(ps->cli, xsphs);

    //cl_error_check= Buffer<float4>(ps->cli, error_check);

    //TODO make a helper constructor for buffer to make a cl_mem from a struct
    std::vector<SPHParams> vparams(0);
    vparams.push_back(params);
    cl_SPHParams = Buffer<SPHParams>(ps->cli, vparams);

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


}

void SPH::setupDomain()
{
    grid.calculateCells(sph_settings.smoothing_distance / sph_settings.simulation_scale);


	grid_params.grid_min = grid.getMin();
	grid_params.grid_max = grid.getMax();
	grid_params.bnd_min  = grid.getBndMin();
	grid_params.bnd_max  = grid.getBndMax();
	grid_params.grid_res = grid.getRes();
	grid_params.grid_size = grid.getSize();
	grid_params.grid_delta = grid.getDelta();
	grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

    printf("gp nb_cells: %d\n", grid_params.nb_cells);


    /*
	grid_params.grid_inv_delta.x = 1. / grid_params.grid_delta.x;
	grid_params.grid_inv_delta.y = 1. / grid_params.grid_delta.y;
	grid_params.grid_inv_delta.z = 1. / grid_params.grid_delta.z;
	grid_params.grid_inv_delta.w = 1.;
    */

    float ss = sph_settings.simulation_scale;

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

int SPH::addBox(int nn, float4 min, float4 max, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = sph_settings.simulation_scale;
    }
    vector<float4> rect = addRect(nn, min, max, sph_settings.spacing, scale);
    pushParticles(rect);
    return rect.size();
}

void SPH::addBall(int nn, float4 center, float radius, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = sph_settings.simulation_scale;
    }
    vector<float4> sphere = addSphere(nn, center, radius, sph_settings.spacing, scale);
    pushParticles(sphere);
}

void SPH::pushParticles(vector<float4> pos)
{
    int nn = pos.size();
    if (num + nn > max_num) {return;}
    float rr = (rand() % 255)/255.0f;
    float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
    //printf("random: %f\n", rr);
	//float4 color(0.0f,0.0f,0.1f,0.1f);

    std::vector<float4> cols(nn);
    std::vector<float4> vels(nn);

    std::fill(cols.begin(), cols.end(),color);
    //float v = .5f;
    float v = 0.0f;
    //float4 iv = float4(v, v, -v, 0.0f);
    float4 iv = float4(0, v, -.1, 0.0f);
    std::fill(vels.begin(), vels.end(),iv);

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

    params.num = num+nn;
    std::vector<SPHParams> vparams(0);
    vparams.push_back(params);
    cl_SPHParams.copyToDevice(vparams);


    printf("about to updateNum\n");
    ps->updateNum(params.num);

    num += nn;  //keep track of number of particles we use

    cl_position.acquire();
    //reprep the unsorted (packed) array to account for new particles
    //might need to do it conditionally if particles are added or subtracted
    printf("about to prep\n");
    prep(1);
    printf("done with prep\n");
    cl_position.release();
#else
    num += nn;  //keep track of number of particles we use
#endif
	renderer->setNum(num);
}

void SPH::render()
{
	System::render();
	renderer->render_box(grid.getBndMin(), grid.getBndMax());
    //renderer->render_table(grid.getBndMin(), grid.getBndMax());
}



} //end namespace
