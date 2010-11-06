
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "SPH.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

#include "oclSortingNetworks_common.h"
extern "C" void closeBitonicSort(void);

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

    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);
    
    //init sph stuff
    //sph_settings.simulation_scale = .001;
    sph_settings.simulation_scale = .001f;
    float scale = sph_settings.simulation_scale;

    //grid = Domain(float4(0,0,0,0), float4(.25/scale, .5/scale, .5/scale, 0));
    //grid = Domain(float4(0,0,0,0), float4(1/scale, 1/scale, 1/scale, 0));
    //grid = Domain(float4(0,0,0,0), float4(1/scale, 1/scale, 1/scale, 0));
    //grid = Domain(float4(0,0,0,0), float4(30, 30, 30, 0));
    grid = Domain(float4(-560,-30,0,0), float4(256, 256, 1276, 0));

    //SPH settings depend on number of particles used
    calculateSPHSettings();
    //set up the grid
    setupDomain();



    sph_settings.integrator = LEAPFROG;
    //sph_settings.integrator = EULER;

    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);
    std::fill(xsphs.begin(), xsphs.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

    //*** end Initialization


#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");

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

    //pure opencl buffers
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

    //setup the sorted and unsorted arrays
    prepareSorted();



    //replace these with the 2 steps
    loadDensity();
    loadPressure();
    loadViscosity();
    loadXSPH();

    //loadStep1();
    //loadStep2();

    loadCollision_wall();

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
    int nn = 2048;
    //float4 min = float4(.4, .4, .1, 0.0f);
    //float4 max = float4(.6, .6, .4, 0.0f);

	float4 min   = float4(-559., -15., .5, 1.);
	float4 max   = float4(220., 225., 450., 1);

    //float4 min = float4(.1, .1, .1, 0.0f);
    //float4 max = float4(.3, .3, .4, 0.0f);

    addBox(nn, min, max, false);
    
    /*
    min = float4(.05/scale, .05/scale, .3/scale, 0.0f);
    max = float4(.2/scale, .2/scale, .45/scale, 0.0f);
    addBox(nn, min, max);
    */

    //float4 center = float4(.1/scale, .15/scale, .3/scale, 0.0f);
    //addBall(nn, center, .06/scale);
    //addBall(nn, center, .1/scale);
    ////////////////// Done with setup particles

    ////DEBUG STUFF
    printf("positions 0: \n");
    positions[0].print();
    
    //////////////


    
    
#endif

}

SPH::~SPH()
{
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

    //Needed while bitonic sort is still C interface
    closeBitonicSort();
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
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
    
    //sub-intervals
    //for(int i=0; i < 10; i++)
    {
        /*
        k_density.execute(num);
        k_pressure.execute(num);
        k_viscosity.execute(num);
        k_xsph.execute(num);
        */
        printf("hash\n");
        hash();
        printf("bitonic_sort\n");
        bitonic_sort();
        printf("data structures\n");
        buildDataStructures(); //reorder
        
        printf("density\n");
        neighborSearch(0);  //density
        printf("forces\n");
        neighborSearch(1);  //forces
        //exit(0);

        printf("collision\n");
        collision();
        printf("integrate\n");
        integrate();
        //exit(0);
    }

    cl_position.release();
    cl_color.release();


}

void SPH::collision()
{
    //when implemented other collision routines can be chosen here
    k_collision_wall.execute(num);
}

void SPH::integrate()
{
    if(sph_settings.integrator == EULER)
    {
        //k_euler.execute(max_num);
        k_euler.execute(num);
    }
    else if(sph_settings.integrator == LEAPFROG)
    {
       //k_leapfrog.execute(max_num);
       k_leapfrog.execute(num);
    }
}


void SPH::calculateSPHSettings()
{
    /*!
    * The Particle Mass (and hence everything following) depends on the MAXIMUM number of particles in the system
    */
    sph_settings.rest_density = 1000;

    sph_settings.particle_mass = (128*1024.0)/max_num * .0002;
    printf("particle mass: %f\n", sph_settings.particle_mass);
    //constant .87 is magic
    sph_settings.particle_rest_distance = .87 * pow(sph_settings.particle_mass / sph_settings.rest_density, 1./3.);
    printf("particle rest distance: %f\n", sph_settings.particle_rest_distance);
    
    //messing with smoothing distance, making it really small to remove interaction still results in weird force values
    sph_settings.smoothing_distance = 2.0f * sph_settings.particle_rest_distance;
    sph_settings.boundary_distance = .5f * sph_settings.particle_rest_distance;

    sph_settings.spacing = sph_settings.particle_rest_distance/ sph_settings.simulation_scale;

    float particle_radius = sph_settings.spacing;
    printf("particle radius: %f\n", particle_radius);
 
    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.simulation_scale = sph_settings.simulation_scale;
    params.boundary_stiffness = 10000.0f;
    params.boundary_dampening = 256.0f;
    params.boundary_distance = sph_settings.particle_rest_distance * .5f;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 20.0f;
    params.num = num;

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

    std::vector<float4> unsorted(max_num*nb_var);
    std::vector<float4> sorted(max_num*nb_var);

    std::fill(unsorted.begin(), unsorted.end(),float4(0.0f, 0.2f, 0.0f, 1.0f));
    std::fill(sorted.begin(), sorted.end(),float4(0.0f, 0.0f, 0.2f, 1.0f));

    //This really should be done on the GPU
    //we probably need to recopy if dynamically adding/removing particles
    /*
	for (int i=0; i < max_num; i++) 
    {
		//vars[i+DENS*num] = densities[i];
		// PROBLEM: density is float, but vars_unsorted is float4
		// HOW TO DEAL WITH THIS WITHOUT DOUBLING MEMORY ACCESS in 
		// buildDataStructures. 

		unsorted[i+DENS*max_num].x = densities[i];
		unsorted[i+DENS*max_num].y = 1.0; // for surface tension (always 1)
		unsorted[i+POS*max_num] = positions[i];
		unsorted[i+VEL*max_num] = velocities[i];
		unsorted[i+FOR*max_num] = forces[i];

		// SHOULD NOT BE REQUIRED
		sorted[i+DENS*max_num].x = densities[i];
		sorted[i+DENS*max_num].y = 1.0;  // for surface tension (always 1)
		sorted[i+POS*max_num] = positions[i];
		sorted[i+VEL*max_num] = velocities[i];
		sorted[i+FOR*max_num] = forces[i];
	}
    */
    cl_vars_unsorted = Buffer<float4>(ps->cli, unsorted);
    cl_vars_sorted = Buffer<float4>(ps->cli, sorted);

    std::vector<int> keys(max_num);
    std::fill(keys.begin(), keys.end(), 0);
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

void SPH::addBox(int nn, float4 min, float4 max, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = sph_settings.simulation_scale;
    }
    vector<float4> rect = addRect(nn, min, max, sph_settings.spacing, scale);
    pushParticles(rect);
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
    float rr = (rand() % 255)/255.0f;
    float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
    printf("random: %f\n", rr);

    std::vector<float4> cols(nn);
    //there should be a better/faster way to do this with vector iterator or something?
    //according to docs the assign function drops previous values which is no good
    for(int i = 0; i < nn; i++)
    {
        positions[num+i] = pos[i];
        cols[i] = color;
    }
#ifdef GPU
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
    
    cl_position.copyToDevice(pos, num);
    cl_color.copyToDevice(cols, num);

    params.num = num+nn;
    std::vector<SPHParams> vparams(0);
    vparams.push_back(params);
    cl_SPHParams.copyToDevice(vparams);

    cl_color.release();
    cl_position.release();

    printf("about to updateNum\n");
    ps->updateNum(params.num);

    num += nn;  //keep track of number of particles we use

    cl_position.acquire();
    //reprep the unsorted (packed) array to account for new particles
    //might need to do it conditionally if particles are added or subtracted
    printf("about to prep\n");
    prep();
    printf("done with prep\n");
    cl_position.release();

#else
    num += nn;  //keep track of number of particles we use
#endif
}



} //end namespace
