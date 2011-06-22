
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "FLOCK.h"
#include "Domain.h"
#include "IV.h"

#include "common/Hose.h"

//for random
#include<time.h>

namespace rtps
{
//using namespace flock;

//----------------------------------------------------------------------
FLOCK::FLOCK(RTPS *psfr, int n)
{
    //store the particle system framework
    ps = psfr;
    settings = ps->settings;
    max_num = n;
    num = 0;
    nb_var = 10;

    resource_path = ps->settings->GetSettingAs<std::string>("rtps_path");
    printf("resource path: %s\n", resource_path.c_str());

    //seed random
    srand ( time(NULL) );

    /*
    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);
    veleval.resize(max_num);
    densities.resize(max_num);
    xflocks.resize(max_num);
    */

    grid = settings->grid;

    std::vector<FLOCKParameters> vparams(0);
    vparams.push_back(flock_params);
    cl_FLOCKParameters= Buffer<FLOCKParameters>(ps->cli, vparams);



    //FLOCKSettings depend on number of particles used
	// Must be called before load kernel methods!
    //calculateFLOCKSettings();
    
    calculate();
    updateFLOCKP();

    spacing = settings->GetSettingAs<float>("Spacing");

    //set up the grid
    setupDomain();
    
    //*** end Initialization

    setupTimers();

#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");

    //setup the sorted and unsorted arrays
    prepareSorted();

    //should be more cross platform
    flock_source_dir = resource_path + "/" + std::string(FLOCK_CL_SOURCE_DIR);
    common_source_dir = resource_path + "/" + std::string(COMMON_CL_SOURCE_DIR);

    ps->cli->addIncludeDir(flock_source_dir);
    ps->cli->addIncludeDir(common_source_dir);


    //std::string cl_includes(FLOCK_CL_SOURCE_DIR);
    //ps->cli->addIncludeDir(cl_includes);



    //loadEuler();
    //loadScopy();
    //loadPrep();
    //loadHash();
    //loadBitonicSort();
    //loadDataStructures();
    //loadNeighbors();
    
    //prep = Prep(ps->cli, timers["prep_gpu"]);
    //hash = Hash(ps->cli, timers["hash_gpu"]);
    //bitonic = Bitonic<unsigned int>(ps->cli);
    //cellindices = CellIndices(ps->cli, timers["ci_gpu"]);
    //permute = Permute(ps->cli, timers["perm_gpu"]);
    
    hash = Hash(common_source_dir, ps->cli, timers["hash_gpu"]);
    bitonic = Bitonic<unsigned int>(common_source_dir, ps->cli );
    cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
    permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );
    
    //computeRules = ComputeRules(flock_source_dir, ps->cli, timers["computeRules_gpu"]);
    //averageRules = AverageRules(flock_source_dir, ps->cli, timers["averageRules_gpu"]);
    rules = Rules(flock_source_dir, ps->cli, timers["rules_gpu"]);
    euler_integration = EulerIntegration(flock_source_dir, ps->cli, timers["euler_gpu"]);


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

    Hose* hose;
    int hs = hoses.size();  
    for(int i = 0; i < hs; i++)
    {
        hose = hoses[i];
        delete hose;
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
    //timers[TI_UPDATE]->start();
    
    //cpuComputeRules();  // based on my boids program
    //cpuAverageRules();
    cpuRules();

    // mymese debugging
#if 0
    for(int i = 0; i < num; i+=64)
    {
        printf("particle %d, positions: %f %f %f  \n", positions[i].x, positions[i].y, positions[i].z);
    }
#endif

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    
    //timers[TI_UPDATE]->end();
}

//----------------------------------------------------------------------
void FLOCK::updateGPU()
{
#if 0 
    //mymese debbug
    printf("smoth_dist: %f\n", flock_params.smoothing_distance);
    printf("radius: %f\n", flock_params.search_radius);
    printf("min dist: %f \n", flock_params.min_dist);
#endif

    //timers[TI_UPDATE]->start();
    timers["update"]->start();
    glFinish();

    if(settings->has_changed())
        updateFLOCKP();

    //sub-intervals
    int sub_intervals = 1;  //should be a setting
    
    for (int i=0; i < sub_intervals; i++)
    {
        sprayHoses();
    }

    cl_position_u.acquire();
    cl_color_u.acquire();
    
    for(int i=0; i < sub_intervals; i++)
    {
        /*
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
        */

        hash_and_sort();


        timers["cellindices"]->start();
        int nc = cellindices.execute(   num,
            cl_sort_hashes,
            cl_sort_indices,
            cl_cell_indices_start,
            cl_cell_indices_end,
            //cl_FLOCKParameters,
            cl_GridParams,
            grid_params.nb_cells,
            clf_debug,
            cli_debug);
        timers["cellindices"]->stop();
       
        timers["permute"]->start();
        permute.execute(   num,
            cl_position_u,
            cl_position_s,
            cl_velocity_u,
            cl_velocity_s,
            cl_veleval_u,
            cl_veleval_s,
            cl_color_u,
            cl_color_s,
            cl_sort_indices,
            //cl_FLOCKParameters,
            cl_GridParams,
            clf_debug,
            cli_debug);
        timers["permute"]->stop();

            //printf("num %d, nc %d\n", num, nc);
        if (nc <= num && nc >= 0)
        {
            //check if the number of particles have changed
            //(this happens when particles go out of bounds,
            //  either because of forces or by explicitly placing
            //  them in order to delete)
            //
            //if so we need to copy sorted into unsorted
            //and redo hash_and_sort
            printf("SOME PARTICLES WERE DELETED!\n");
            printf("nc: %d num: %d\n", nc, num);

            deleted_pos.resize(num-nc);
            deleted_vel.resize(num-nc);
            //The deleted particles should be the nc particles after num
            cl_position_s.copyToHost(deleted_pos, nc); //damn these will always be out of bounds here!
            cl_velocity_s.copyToHost(deleted_vel, nc);

 
            num = nc;
            settings->SetSetting("Number of Particles", num);
            //sphp.num = num;
            updateFLOCKP();
            renderer->setNum(flock_params.num);
            //need to copy sorted arrays into unsorted arrays
            call_prep(2);
            //printf("HOW MANY NOW? %d\n", num);
            hash_and_sort();
            //we've changed num and copied sorted to unsorted. skip this iteration and do next one
            //this doesn't work because sorted force etc. are having an effect?
            //continue; 
        }

        //mymese debbug
        #if 0 
            flock_params.smoothing_distance = 333.;
            flock_params.search_radius = 222.;
            flock_params.min_dist = 111.;

            std::vector<FLOCKParameters> vparams(0);
            vparams.push_back(flock_params);
            cl_FLOCKParameters.copyToDevice(vparams);
        #endif

        
      /*  timers["computeRules"]->start();
        computeRules.execute(   num,
            //cl_vars_sorted,
            cl_position_s,
            cl_velocity_s,
            cl_separation_s,
            cl_alignment_s,
            cl_cohesion_s,
            cl_flockmates_s,
            cl_cell_indices_start,
            cl_cell_indices_end,
            cl_GridParamsScaled,
            cl_FLOCKParameters,
            clf_debug,
            cli_debug);
       timers["computeRules"]->stop();*/
            
        timers["rules"]->start();
        // add a bool variable for each rule
        if(1){
            rules.executeFlockmates(   num,
                cl_position_s,
                cl_flockmates_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_GridParamsScaled,
                cl_FLOCKParameters,
                clf_debug,
                cli_debug);
        }
        if(1){
            rules.executeSeparation(   num,
                cl_position_s,
                cl_separation_s,
                cl_flockmates_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_GridParamsScaled,
                cl_FLOCKParameters,
                clf_debug,
                cli_debug);
        }
        if(1){
            rules.executeAlignment(   num,
                cl_position_s,
                cl_velocity_s,
                cl_alignment_s,
                cl_flockmates_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_GridParamsScaled,
                cl_FLOCKParameters,
                clf_debug,
                cli_debug);
        }
        if(1){
            rules.executeCohesion(   num,
                cl_position_s,
                cl_cohesion_s,
                cl_flockmates_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_GridParamsScaled,
                cl_FLOCKParameters,
                clf_debug,
                cli_debug);
        }
        timers["rules"]->stop();
        
        //collision();
        
        timers["integrate"]->start();
        integrate();
        timers["integrate"]->stop();

    }

    cl_position_u.release();
    cl_color_u.release();

    //timers[TI_UPDATE]->end();
    timers["update"]->stop();
}

//----------------------------------------------------------------------
void FLOCK::hash_and_sort()
{
    //printf("hash\n");
    timers["hash"]->start();
    hash.execute(   num,
        cl_position_u,
        cl_sort_hashes,
        cl_sort_indices,
        cl_GridParams,
        clf_debug,
        cli_debug);
    timers["hash"]->stop();

    //printf("bitonic_sort\n");
    //defined in Sort.cpp
    timers["bitonic"]->start();
    bitonic_sort();
    timers["bitonic"]->stop();
}

//----------------------------------------------------------------------
void FLOCK::integrate()
{
    euler_integration.execute(num,
        settings->dt,
        cl_position_u,
        cl_position_s,
        cl_velocity_u,
        cl_velocity_s,
        cl_separation_s,
        cl_alignment_s,
        cl_cohesion_s,
        cl_sort_indices,
        cl_FLOCKParameters,
        cl_GridParamsScaled,
        //debug
        clf_debug,
        cli_debug);


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
void FLOCK::call_prep(int stage)
{
    //Replace with enqueueCopyBuffer
/*    prep.execute(num,
        stage,
        cl_position_u,
        cl_position_s,
        cl_velocity_u,
        cl_velocity_s,
        cl_veleval_u,
        cl_veleval_s,
        cl_color_u,
        cl_color_s,
        //cl_vars_unsorted, 
        //cl_vars_sorted, 
        cl_sort_indices,
        //params
        cl_FLOCKParameters,
        //Buffer<GridParams>& gp,
        //debug params
        clf_debug,
        cli_debug);*/


        //Replace with enqueueCopyBuffer

        cl_position_u.copyFromBuffer(cl_position_s, 0, 0, num);
        cl_velocity_u.copyFromBuffer(cl_velocity_s, 0, 0, num);
        cl_veleval_u.copyFromBuffer(cl_veleval_s, 0, 0, num);
        cl_color_u.copyFromBuffer(cl_color_s, 0, 0, num);

}

//----------------------------------------------------------------------
int FLOCK::setupTimers()
{
    int print_freq = 1000; //one second
    int time_offset = 5;
/*
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
*/

    timers["update"] = new EB::Timer("Update loop", time_offset);
    timers["hash"] = new EB::Timer("Hash function", time_offset);
    timers["hash_gpu"] = new EB::Timer("Hash GPU kernel execution", time_offset);
    //timers["datastructures"] = new EB::Timer("Datastructures function", time_offset);
    timers["cellindices"] = new EB::Timer("CellIndices function", time_offset);
    timers["ci_gpu"] = new EB::Timer("CellIndices GPU kernel execution", time_offset);
    timers["permute"] = new EB::Timer("Permute function", time_offset);
    timers["perm_gpu"] = new EB::Timer("Permute GPU kernel execution", time_offset);
    timers["ds_gpu"] = new EB::Timer("DataStructures GPU kernel execution", time_offset);
    timers["bitonic"] = new EB::Timer("Bitonic Sort function", time_offset);
    //timers["neighbor"] = new EB::Timer("Neighbor Total", time_offset);
    timers["computeRules"] = new EB::Timer("Compute Rules function", time_offset);
    timers["computeRules_gpu"] = new EB::Timer("Compute Rules GPU kernel execution", time_offset);
    //timers["collision_wall"] = new EB::Timer("Collision wall function", time_offset);
    //timers["cw_gpu"] = new EB::Timer("Collision Wall GPU kernel execution", time_offset);
    //timers["collision_tri"] = new EB::Timer("Collision triangles function", time_offset);
    //timers["ct_gpu"] = new EB::Timer("Collision Triangle GPU kernel execution", time_offset);
    timers["integrate"] = new EB::Timer("Integration kernel execution", time_offset);
    timers["euler_gpu"] = new EB::Timer("Euler integration", time_offset);
    timers["averageRules"] = new EB::Timer("Average Rules function", time_offset);
    timers["averageRules_gpu"] = new EB::Timer("Average Rules GPU kernel execution", time_offset);
    //timers["prep_gpu"] = new EB::Timer("Prep GPU kernel execution", time_offset);
    timers["rules"] = new EB::Timer("Computes all the rules", time_offset);
    timers["rules_gpu"] = new EB::Timer("Computes all the rules in the GPU", time_offset);

	return 0;
}

//----------------------------------------------------------------------
void FLOCK::printTimers()
{
//    for(int i = 0; i < 11; i++) //switch to vector of timers and use size()
//    {
//        timers[i]->print();
//    }
//    System::printTimers();
    timers.printAll();
    timers.writeToFile("flock_timer_log");
}

//----------------------------------------------------------------------
/*void FLOCK::calculateFLOCKSettings()
{

    float4 dmin = grid->getBndMin();
    float4 dmax = grid->getBndMax();

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
*/
//----------------------------------------------------------------------
void FLOCK::prepareSorted()
{
//    #include "flock/cl_src/cl_macros.h"
 
    positions.resize(max_num);
    velocities.resize(max_num);
    veleval.resize(max_num);
    colors.resize(max_num);
    separation.resize(max_num);
    alignment.resize(max_num);
    cohesion.resize(max_num);
    flockmates.resize(max_num);
    
    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);

    std::fill(velocities.begin(), velocities.end(), float4(0.0f, 0.0f, 0.0f, 0.f));
    std::fill(veleval.begin(), veleval.end(), float4(0.0f, 0.0f, 0.0f, 0.f));
    
    std::fill(separation.begin(), separation.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(alignment.begin(), alignment.end(), float4(0.0f, 0.f, 0.f, 0.f));
    std::fill(cohesion.begin(), cohesion.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(flockmates.begin(), flockmates.end(),int4(0, 0, 0, 0));
    
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
    cl_position_u = Buffer<float4>(ps->cli, pos_vbo);
    cl_position_s = Buffer<float4>(ps->cli, positions);
    cl_color_u = Buffer<float4>(ps->cli, col_vbo);
    cl_color_s = Buffer<float4>(ps->cli, colors);

    //pure opencl buffers: these are deprecated
    cl_velocity_u = Buffer<float4>(ps->cli, velocities);
    cl_velocity_s = Buffer<float4>(ps->cli, velocities);
    cl_veleval_u = Buffer<float4>(ps->cli, veleval);
    cl_veleval_s = Buffer<float4>(ps->cli, veleval);
    
    cl_separation_s = Buffer<float4>(ps->cli, separation);
    cl_alignment_s = Buffer<float4>(ps->cli, alignment);
    cl_cohesion_s = Buffer<float4>(ps->cli, cohesion);
    cl_flockmates_s= Buffer<int4>(ps->cli, flockmates);

    // FLOCK Parameters
    //std::vector<FLOCKParameters> vparams(0);
    //vparams.push_back(flock_params);
    //cl_FLOCKParameters = Buffer<FLOCKParameters>(ps->cli, vparams);
    
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
    std::fill(cliv.begin(), cliv.end(),int4(0, 0, 0, 0));
    clf_debug = Buffer<float4>(ps->cli, clfv);
    cli_debug = Buffer<int4>(ps->cli, cliv);

    /*
    //sorted and unsorted arrays
    std::vector<float4> unsorted(max_num*nb_var);
    std::vector<float4> sorted(max_num*nb_var);

    std::fill(unsorted.begin(), unsorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
    std::fill(sorted.begin(), sorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));

    cl_vars_unsorted = Buffer<float4>(ps->cli, unsorted);
    cl_vars_sorted = Buffer<float4>(ps->cli, sorted);
    */

    std::vector<unsigned int> keys(max_num);
    
    //to get around limits of bitonic sort only handling powers of 2
    #include "limits.h"
    
    std::fill(keys.begin(), keys.end(), INT_MAX);
	cl_sort_indices  = Buffer<unsigned int>(ps->cli, keys);
	cl_sort_hashes   = Buffer<unsigned int>(ps->cli, keys);

    // Size is the grid size. That is a problem since the number of
	// occupied cells could be much less than the number of grid elements. 
    std::vector<unsigned int> gcells(grid_params.nb_cells+1);
	int minus = 0xffffffff;
    std::fill(gcells.begin(), gcells.end(), 666);

	cl_cell_indices_start = Buffer<unsigned int>(ps->cli, gcells);
	cl_cell_indices_end   = Buffer<unsigned int>(ps->cli, gcells);

	// For bitonic sort. Remove when bitonic sort no longer used
	// Currently, there is an error in the Radix Sort (just run both
	// sorts and compare outputs visually
	cl_sort_output_hashes = Buffer<unsigned int>(ps->cli, keys);
	cl_sort_output_indices = Buffer<unsigned int>(ps->cli, keys);

#endif

}

//----------------------------------------------------------------------
void FLOCK::setupDomain()
{
    grid->calculateCells(flock_params.smoothing_distance / flock_params.simulation_scale);

	grid_params.grid_min = grid->getMin();
	grid_params.grid_max = grid->getMax();
	grid_params.bnd_min  = grid->getBndMin();
	grid_params.bnd_max  = grid->getBndMax();
	grid_params.grid_res = grid->getRes();
	grid_params.grid_size = grid->getSize();
	grid_params.grid_delta = grid->getDelta();
	grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

    //printf("gp nb_cells: %d\n", grid_params.nb_cells);

    // debug mymese
#if 0
    float4 gmin = grid_params.bnd_min;
    float4 gmax = grid_params.bnd_max;
    float4 bd = grid_params.grid_size;
    printf("\n *************** \n grid size: %f, %f, %f\n", bd.x, bd.y, bd.z); 
    printf("min boundary: %f, %f, %f\n", gmin.x, gmin.y, gmin.z);
    printf("max boundary: %f, %f, %f\n ************** \n", gmax.x,gmax.y, gmax.z);
#endif

    float ss = flock_params.simulation_scale;

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
        scale = flock_params.simulation_scale;
    }
    
    printf("\n\n ADDING A CUBE \n\n");
    
    vector<float4> rect;
    addCube(nn, min, max, spacing, scale, rect);
    
    float4 velo(0.f,0.f,0.f,0.f);
    pushParticles(rect, velo, color);  // BLENDER
    
    return rect.size();
}

//----------------------------------------------------------------------
void FLOCK::addBall(int nn, float4 center, float radius, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_params.simulation_scale;
    }
    
    printf("\n\n ADDING A SPHERE \n\n");
    
    float4 velo(0.f,0.f,0.f,0.f);
    float4 color(255.f,0.f,0.f,0.f);
    vector<float4> sphere = addSphere(nn, center, radius, spacing, scale);
    pushParticles(sphere, velo, color);
}

//----------------------------------------------------------------------
int FLOCK::addHose(int total_n, float4 center, float4 velocity, float radius, float4 color)
{
    //in sph we just use sph spacing
    radius *= spacing;
    Hose* hose = new Hose(ps, total_n, center, velocity, radius, spacing, color);
    hoses.push_back(hose);
    //printf("size of hoses: %d\n", hoses.size());
    return hoses.size()-1;

}

//----------------------------------------------------------------------
void FLOCK::updateHose(int index, float4 center, float4 velocity, float radius, float4 color)
{
    //we need to expose the vector of hoses somehow
    //doesn't seem right to make user manage an index
    //in sph we just use sph spacing
    radius *= spacing;
    hoses[index]->update(center, velocity, radius, spacing, color);
    //printf("size of hoses: %d\n", hoses.size());
}

//----------------------------------------------------------------------
void FLOCK::sprayHoses()
{
    std::vector<float4> parts;
    for (int i = 0; i < hoses.size(); i++)
    {
        parts = hoses[i]->spray();
        
        if (parts.size() > 0)
            pushParticles(parts, hoses[i]->getVelocity(), hoses[i]->getColor());
    }
}

//----------------------------------------------------------------------
void FLOCK::testDelete()
{
    //cut = 1;
    std::vector<float4> poss(40);
    float4 posx(100.,100.,100.,1.);
    std::fill(poss.begin(), poss.end(),posx);
    //cl_vars_unsorted.copyToDevice(poss, max_num + 2);
    cl_position_u.acquire();
    cl_position_u.copyToDevice(poss);
    cl_position_u.release();
    ps->cli->queue[0].finish();
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
    
    std::fill(cols.begin(), cols.end(),color); //BLENDER
    //float4 iv = float4(0.f, 0.f, 0.f, 0.0f);   
    //std::fill(vels.begin(), vels.end(),iv);   
    //printf("PUSH PARTICLES\n");

#ifdef CPU
    std::copy(pos.begin(), pos.end(), positions.begin()+num);
#endif

#ifdef GPU
    glFinish();
    cl_position_u.acquire();
    cl_color_u.acquire();
 
    //prep(0);

    cl_position_u.copyToDevice(pos, num);
    cl_color_u.copyToDevice(cols, num);
    cl_velocity_u.copyToDevice(vels, num);

    settings->SetSetting("Number of Particles", num+nn);
    updateFLOCKP();

    num += nn;  //keep track of number of particles we use
    
    cl_color_u.release();
    cl_position_u.release();

    //flock_params.num = num+nn;
    //updateFLOCKP();



    //cl_position.acquire();
    
    //reprep the unsorted (packed) array to account for new particles
    //might need to do it conditionally if particles are added or subtracted
    //prep(1);
    //cl_position.release();

#else
    num += nn;  //keep track of number of particles we use
#endif

	renderer->setNum(num);
}

//----------------------------------------------------------------------
/*
void FLOCK::updateFLOCKP()
{
    std::vector<FLOCKParameters> vparams(0);
    vparams.push_back(flock_params);
    cl_FLOCKParameters.copyToDevice(vparams);
}
*/
//----------------------------------------------------------------------
void FLOCK::render()
{
    renderer->render_box(grid->getBndMin(), grid->getBndMax());
    //renderer->render_table(grid->getBndMin(), grid->getBndMax());
    System::render();
}

//----------------------------------------------------------------------
void FLOCK::setRenderer()
{
    switch(ps->settings->getRenderType())
    {
        case RTPSettings::SPRITE_RENDER:
            renderer = new SpriteRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            //printf("spacing for radius %f\n", spacing);
            break;
        case RTPSettings::SCREEN_SPACE_RENDER:
            renderer = new SSFRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        case RTPSettings::RENDER:
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        case RTPSettings::MESH_RENDER:
            renderer = new MeshRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        default:
            //should be an error
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
        break;
    }
    renderer->setParticleRadius(spacing);
}


} //end namespace
