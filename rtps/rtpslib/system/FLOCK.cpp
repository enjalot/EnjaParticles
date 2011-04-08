
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "FLOCK.h"
#include "Domain.h"
#include "IV.h"

#include "common/Hose.h"

//for random
#include<time.h>

namespace rtps {

//----------------------------------------------------------------------
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

    //FLOCKSettings depend on number of particles used
	// Must be called before load kernel methods!
    calculateFLOCKSettings();
    
    //set up the grid
    setupDomain();
    
    //*** end Initialization

    setupTimers();

    //setup the sorted and unsorted arrays
    prepareSorted();

    std::string cl_includes(FLOCK_CL_SOURCE_DIR);
    ps->cli->setIncludeDir(cl_includes);

#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");

    loadEuler();

    loadScopy();

    loadPrep();
    loadHash();
    loadBitonicSort();
    loadDataStructures();
    loadNeighbors();
    
#endif
     
    setRenderer();

}

//----------------------------------------------------------------------
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

//----------------------------------------------------------------------
void FLOCK::update()
{
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
    timers[TI_UPDATE]->start();
    
    ge_cpuEuler();  // based on my boids program

    // mymese debugging
#if 0
    for(int i = 0; i < num; i+=64)
    {
        printf("particle %d, positions: %f %f %f  \n", positions[i].x, positions[i].y, positions[i].z);
    }
#endif

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    
    timers[TI_UPDATE]->end();
}

//----------------------------------------------------------------------
void FLOCK::updateGPU()
{

    timers[TI_UPDATE]->start();
    glFinish();

    
    //sub-intervals
    int sub_intervals = 1;  //should be a setting
    
    for (int i=0; i < sub_intervals; i++)
    {
        sprayHoses();
    }

    cl_position.acquire();
    cl_color.acquire();
    
    for(int i=0; i < sub_intervals; i++)
    {
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
        
        // neighbor list
        timers[TI_NEIGH]->start();

        //printf("density\n");
        timers[TI_DENS]->start();
        neighborSearch(0);  //density => flockmates
        timers[TI_DENS]->end();

        timers[TI_NEIGH]->end();
        
        //printf("integrate\n");
        integrate();		// compute the rules and itegrate
    }

    cl_position.release();
    cl_color.release();

    timers[TI_UPDATE]->end();

}

//----------------------------------------------------------------------
void FLOCK::integrate()
{
    int local_size = 128;
        
    timers[TI_EULER]->start();
    k_euler.execute(num, local_size);
    timers[TI_EULER]->end();

    // mymese debugging
#if 0 
    if(num > 0)
    {
        std::vector<int4> cli(num);
        cli_debug.copyToHost(cli);

        std::vector<float4> clf(num);
        clf_debug.copyToHost(clf);

        for(int i = 0; i < 4; i++)
        {
            printf("numFlockmates = %d and count = %d \n", cli[i].x, cli[i].y);
            printf("clf[%d] = %f %f %f %f\n", i, clf[i].x, clf[i].y, clf[i].z, clf[i].w);
        }
		printf("num= %d\n", num);
        printf("\n\n");
    }
#endif

}

//----------------------------------------------------------------------
int FLOCK::setupTimers()
{
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

	return 0;
}

//----------------------------------------------------------------------
void FLOCK::printTimers()
{
    for(int i = 0; i < 11; i++) //switch to vector of timers and use size()
    {
        timers[i]->print();
    }
    System::printTimers();
}

//----------------------------------------------------------------------
void FLOCK::calculateFLOCKSettings()
{

    float4 dmin = grid.getBndMin();
    float4 dmax = grid.getBndMax();

    //constant .87 is magic
    //flock_settings.particle_rest_distance = .87 * pow(particle_vol, 1./3.);
    flock_settings.particle_rest_distance = .05;
    printf("particle rest distance: %f\n", flock_settings.particle_rest_distance);
    
    //messing with smoothing distance, making it really small to remove interaction still results in weird force values
    //flock_settings.smoothing_distance = .50f;
    flock_settings.smoothing_distance = 2.0f * flock_settings.particle_rest_distance;

    //flock_settings.simulation_scale = pow(particle_vol * max_num / domain_vol, 1./3.); 
    flock_settings.simulation_scale = 1.0f;
    printf("simulation scale: %f\n", flock_settings.simulation_scale);

    //flock_settings.spacing = flock_settings.particle_rest_distance/ flock_settings.simulation_scale;
    flock_settings.spacing = 0.050f; // must be less than smoothing_distance
    
    // FLOCKParameters
    flock_params.grid_min = dmin;
    flock_params.grid_max = dmax;
    flock_params.rest_distance = flock_settings.particle_rest_distance;
    flock_params.smoothing_distance = flock_settings.smoothing_distance;
    flock_params.num = num;
    
    // Boids flock_params
	flock_params.min_dist     = 0.5f * flock_params.smoothing_distance * ps->settings.min_dist; // desired separation between boids
    flock_params.search_radius= 0.8f * flock_params.smoothing_distance * ps->settings.search_radius;
    flock_params.max_speed    = 1.0f * ps->settings.max_speed;

    flock_params.w_sep = ps->settings.w_sep;
    flock_params.w_align = ps->settings.w_align;
    flock_params.w_coh= ps->settings.w_coh;
    
    // debug mymese
#if 0
    float4 gmin = params.grid_min;
    float4 gmax = params.grid_max;
    float bd = params.boundary_distance;
    printf("\n *************** \n boundary distance: %f\n", bd); 
    printf("min grid: %f, %f, %f\n", gmin.x, gmin.y, gmin.z);
    printf("max grid: %f, %f, %f\n ************** \n", gmax.x,gmax.y, gmax.z);
#endif

}

//----------------------------------------------------------------------
void FLOCK::prepareSorted()
{
    #include "flock/cl_src/cl_macros.h"
    
    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);
 
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(), float4(0.0f, 0.0f, 0.0f, 1.f));
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

    // FLOCK Parameters
    std::vector<FLOCKParameters> vparams(0);
    vparams.push_back(flock_params);
    cl_FLOCKParameters = Buffer<FLOCKParameters>(ps->cli, vparams);
    
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

    // Size is the grid size. That is a problem since the number of
	// occupied cells could be much less than the number of grid elements. 
    std::vector<int> gcells(grid_params.nb_cells);
	int minus = 0xffffffff;
    std::fill(gcells.begin(), gcells.end(), 0);

	cl_cell_indices_start = Buffer<int>(ps->cli, gcells);
	cl_cell_indices_end   = Buffer<int>(ps->cli, gcells);

	// For bitonic sort. Remove when bitonic sort no longer used
	// Currently, there is an error in the Radix Sort (just run both
	// sorts and compare outputs visually
	cl_sort_output_hashes = Buffer<int>(ps->cli, keys);
	cl_sort_output_indices = Buffer<int>(ps->cli, keys);

#endif

}

//----------------------------------------------------------------------
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
#if 0
    float4 gmin = grid_params.bnd_min;
    float4 gmax = grid_params.bnd_max;
    float4 bd = grid_params.grid_size;
    printf("\n *************** \n grid size: %f, %f, %f\n", bd.x, bd.y, bd.z); 
    printf("min boundary: %f, %f, %f\n", gmin.x, gmin.y, gmin.z);
    printf("max boundary: %f, %f, %f\n ************** \n", gmax.x,gmax.y, gmax.z);
#endif

    float ss = flock_settings.simulation_scale;

	grid_params_scaled.grid_min = grid_params.grid_min * ss;
	grid_params_scaled.grid_max = grid_params.grid_max * ss;
	grid_params_scaled.bnd_min  = grid_params.bnd_min * ss;
	grid_params_scaled.bnd_max  = grid_params.bnd_max * ss;
	grid_params_scaled.grid_res = grid_params.grid_res;
	grid_params_scaled.grid_size = grid_params.grid_size * ss;
	grid_params_scaled.grid_delta = grid_params.grid_delta / ss;
    grid_params_scaled.nb_cells = grid_params.nb_cells;

    grid_params.print();
    grid_params_scaled.print();
    
}

//----------------------------------------------------------------------
int FLOCK::addBox(int nn, float4 min, float4 max, bool scaled, float4 color)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_settings.simulation_scale;
    }
    
    printf("\n\n ADDING A CUBE \n\n");
    
    vector<float4> rect;
    
    addCube(nn, min, max, flock_settings.spacing, scale, rect);
    pushParticles(rect, float4(0.f,0.f,0.f,0.f), ps->settings.color);
    
    return rect.size();
}

//----------------------------------------------------------------------
void FLOCK::addBall(int nn, float4 center, float radius, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_settings.simulation_scale;
    }
    
    printf("\n\n ADDING A SPHERE \n\n");
    
    vector<float4> flockere = addSphere(nn, center, radius, flock_settings.spacing, scale);
    pushParticles(flockere, float4(0.f,0.f,0.f,0.f), ps->settings.color);
}

//----------------------------------------------------------------------
void FLOCK::addHose(int total_n, float4 center, float4 velocity, float radius, float4 color)
{
    printf("wtf for real\n");
    //in sph we just use sph spacing
    radius *= flock_settings.spacing;
    Hose hose = Hose(ps, total_n, center, velocity, radius, flock_settings.spacing, color);
    printf("wtf\n");
    hoses.push_back(hose);
    printf("size of hoses: %d\n", hoses.size());
}

//----------------------------------------------------------------------
void FLOCK::sprayHoses()
{
    std::vector<float4> parts;
    for (int i = 0; i < hoses.size(); i++)
    {
        parts = hoses[i].spray();
        
        if (parts.size() > 0)
            pushParticles(parts, hoses[i].getVelocity(), hoses[i].getColor());
    }
}

//----------------------------------------------------------------------
void FLOCK::pushParticles(vector<float4> pos, float4 velo, float4 color)
{
    int nn = pos.size();
    std::vector<float4> vels(nn);
    std::fill(vels.begin(), vels.end(), velo);
    pushParticles(pos, vels, color);
}

//----------------------------------------------------------------------
void FLOCK::pushParticles(vector<float4> pos, vector<float4> vels, float4 color)
{
    int nn = pos.size();
    
    // if we have reach max num of particles, then return
    if (num + nn > max_num) {return;}
    
    //float rr = (rand() % 255)/255.0f;
    //float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
    
    std::vector<float4> cols(nn);
    //std::vector<float4> vels(nn);
    
    std::fill(cols.begin(), cols.end(),ps->settings.color);
    //float4 iv = float4(0.f, 0.f, 0.f, 0.0f);   
    //std::fill(vels.begin(), vels.end(),iv);   
printf("pPUSH PARTICLES\n");
#ifdef CPU
 std::copy(pos.begin(), pos.end(), positions.begin()+num);
#endif

#ifdef GPU
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
 
    prep(0);

    cl_position.copyToDevice(pos, num);
    cl_color.copyToDevice(cols, num);

    cl_color.release();
    cl_position.release();

    cl_velocity.copyToDevice(vels, num);

    flock_params.num = num+nn;
    updateFLOCKP();

    num += nn;  //keep track of number of particles we use

    cl_position.acquire();
    
    //reprep the unsorted (packed) array to account for new particles
    //might need to do it conditionally if particles are added or subtracted
    prep(1);
    cl_position.release();

#else
    num += nn;  //keep track of number of particles we use
#endif

	renderer->setNum(num);
}

//----------------------------------------------------------------------
void FLOCK::updateFLOCKP()
{
    std::vector<FLOCKParameters> vparams(0);
    vparams.push_back(flock_params);
    cl_FLOCKParameters.copyToDevice(vparams);
}

//----------------------------------------------------------------------
void FLOCK::render()
{
	System::render();
	renderer->render_box(grid.getBndMin(), grid.getBndMax());
}

//----------------------------------------------------------------------
void FLOCK::setRenderer()
{
    switch(ps->settings.getRenderType())
    {
        case RTPSettings::SPRITE_RENDER:
            renderer = new SpriteRender(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
            printf("spacing for radius %f\n", flock_settings.spacing);
            break;
        case RTPSettings::SCREEN_SPACE_RENDER:
            renderer = new SSFRender(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
            break;
        case RTPSettings::RENDER:
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
            break;
        default:
            //should be an error
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
        break;
    }
    renderer->setParticleRadius(flock_settings.spacing/2);
}


} //end namespace
