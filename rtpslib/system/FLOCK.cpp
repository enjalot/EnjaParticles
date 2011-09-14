/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/



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

//----------------------------------------------------------------------
FLOCK::FLOCK(RTPS *psfr, int n)
{
    //store the particle system framework
    ps = psfr;
    settings = ps->settings;
    max_num = n;
    num = 0;

    resource_path = ps->settings->GetSettingAs<std::string>("rtps_path");
    printf("resource path: %s\n", resource_path.c_str());

    //seed random
    srand ( time(NULL) );

    grid = settings->grid;

    std::vector<FLOCKParameters> vparams(0);
    vparams.push_back(flock_params);
    cl_FLOCKParameters= Buffer<FLOCKParameters>(ps->cli, vparams);

    calculate();
    updateFLOCKP();

    spacing = settings->GetSettingAs<float>("Spacing");

    //set up the grid
    setupDomain();
    
    //set up the timers 
    setupTimers();

    //setup the sorted and unsorted arrays
    prepareSorted();

#ifdef CPU
    printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
    printf("RUNNING ON THE GPU\n");

    //should be more cross platform
    flock_source_dir = resource_path + "/" + std::string(FLOCK_CL_SOURCE_DIR);
    common_source_dir = resource_path + "/" + std::string(COMMON_CL_SOURCE_DIR);

    ps->cli->addIncludeDir(flock_source_dir);
    ps->cli->addIncludeDir(common_source_dir);
    
    hash = Hash(common_source_dir, ps->cli, timers["hash_gpu"]);
    bitonic = Bitonic<unsigned int>(common_source_dir, ps->cli );
    cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
    permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );
    
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
    if(settings->has_changed())
        updateFLOCKP();
    
    cpuRules();
    cpuEulerIntegration();

    // mymese debugging
#if 0
    for(int i = 0; i < num; i+=64)
    {
        printf("particle %d, positions: %f %f %f  \n", positions[i].x, positions[i].y, positions[i].z);
    }
#endif

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    
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
        hash_and_sort();

        timers["cellindices"]->start();
        int nc = cellindices.execute(   num,
            cl_sort_hashes,
            cl_sort_indices,
            cl_cell_indices_start,
            cl_cell_indices_end,
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
            cl_GridParams,
            clf_debug,
            cli_debug);
        timers["permute"]->stop();

        if (nc <= num && nc >= 0)
        {
            printf("SOME PARTICLES WERE DELETED!\n");
            printf("nc: %d num: %d\n", nc, num);

            deleted_pos.resize(num-nc);
            deleted_vel.resize(num-nc);
            
            cl_position_s.copyToHost(deleted_pos, nc); 
            cl_velocity_s.copyToHost(deleted_vel, nc);
 
            num = nc;
            settings->SetSetting("Number of Particles", num);
            
            updateFLOCKP();
            renderer->setNum(flock_params.num);
            
            //need to copy sorted arrays into unsorted arrays
            call_prep(2);
            
            hash_and_sort();
        }

        timers["rules"]->start();

        if(flock_params.w_sep > 0.f || flock_params.w_align > 0.f || flock_params.w_coh > 0.f || flock_params.w_goal > 0.f || flock_params.w_avoid > 0.f){
            rules.execute(   num,
                settings->target,
                cl_position_s,
                cl_velocity_s,
                cl_flockmates_s,
                cl_separation_s,
                cl_alignment_s,
                cl_cohesion_s,
                cl_goal_s,
                cl_avoid_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_GridParamsScaled,
                cl_FLOCKParameters,
                clf_debug,
                cli_debug);
        }
        
        timers["rules"]->stop();
        
        timers["integrate"]->start();
        integrate();
        timers["integrate"]->stop();

    }

    cl_position_u.release();
    cl_color_u.release();

    timers["update"]->stop();
}

//----------------------------------------------------------------------
void FLOCK::hash_and_sort()
{
    timers["hash"]->start();
    hash.execute(   num,
        cl_position_u,
        cl_sort_hashes,
        cl_sort_indices,
        cl_GridParams,
        clf_debug,
        cli_debug);
    timers["hash"]->stop();

    timers["bitonic"]->start();
    bitonic_sort();
    timers["bitonic"]->stop();
}

//----------------------------------------------------------------------
void FLOCK::integrate()
{
    euler_integration.execute(num,
        settings->dt,
        settings->two_dimensional,
        cl_position_u,
        cl_position_s,
        cl_velocity_u,
        cl_velocity_s,
        cl_separation_s,
        cl_alignment_s,
        cl_cohesion_s,
        cl_goal_s,
        cl_avoid_s,
        cl_leaderfollowing_s,
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
            printf("w1 = %d\t w2 = %d\t w3 = %d\t w4 = %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
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

    timers["update"] = new EB::Timer("Update loop", time_offset);
    
    timers["hash"] = new EB::Timer("Hash function", time_offset);
    timers["hash_gpu"] = new EB::Timer("Hash GPU kernel execution", time_offset);
    timers["cellindices"] = new EB::Timer("CellIndices function", time_offset);
    timers["ci_gpu"] = new EB::Timer("CellIndices GPU kernel execution", time_offset);
    timers["permute"] = new EB::Timer("Permute function", time_offset);
    timers["perm_gpu"] = new EB::Timer("Permute GPU kernel execution", time_offset);
    timers["bitonic"] = new EB::Timer("Bitonic Sort function", time_offset);
    
    timers["integrate"] = new EB::Timer("Integration kernel execution", time_offset);
    timers["euler_gpu"] = new EB::Timer("Euler integration", time_offset);
    timers["rules"] = new EB::Timer("Computes all the rules", time_offset);
    timers["rules_gpu"] = new EB::Timer("Computes all the rules in the GPU", time_offset);

	return 0;
}

//----------------------------------------------------------------------
void FLOCK::printTimers()
{
    timers.printAll();
    timers.writeToFile("flock_timer_log");
}

//----------------------------------------------------------------------
void FLOCK::prepareSorted()
{
 
    positions.resize(max_num);
    velocities.resize(max_num);
    veleval.resize(max_num);
    colors.resize(max_num);

    flockmates.resize(max_num);
    separation.resize(max_num);
    alignment.resize(max_num);
    cohesion.resize(max_num);
    goal.resize(max_num);
    avoid.resize(max_num);
    wander.resize(max_num);
    leaderfollowing.resize(max_num);

    //for reading back different values from the kernel
    std::vector<float4> error_check(max_num);

    std::fill(velocities.begin(), velocities.end(), float4(0.0f, 0.0f, 0.0f, 0.f));
    std::fill(veleval.begin(), veleval.end(), float4(0.0f, 0.0f, 0.0f, 0.f));
    
    std::fill(flockmates.begin(), flockmates.end(),int4(0, 0, 0, 0));
    std::fill(separation.begin(), separation.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(alignment.begin(), alignment.end(), float4(0.0f, 0.f, 0.f, 0.f));
    std::fill(cohesion.begin(), cohesion.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(goal.begin(), goal.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(avoid.begin(), avoid.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(wander.begin(), wander.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(leaderfollowing.begin(), leaderfollowing.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    
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
    
    cl_flockmates_s= Buffer<int4>(ps->cli, flockmates);
    cl_separation_s = Buffer<float4>(ps->cli, separation);
    cl_alignment_s = Buffer<float4>(ps->cli, alignment);
    cl_cohesion_s = Buffer<float4>(ps->cli, cohesion);
    cl_goal_s = Buffer<float4>(ps->cli, goal);
    cl_avoid_s = Buffer<float4>(ps->cli, avoid);
    cl_leaderfollowing_s = Buffer<float4>(ps->cli, leaderfollowing);

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
void FLOCK::addBall(int nn, float4 center, float radius, bool scaled, float4 color)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = flock_params.simulation_scale;
    }
    
    printf("\n\n ADDING A SPHERE \n\n");
    
    vector<float4> sphere = addSphere(nn, center, radius, spacing, scale);
    
    float4 velo(0.f,0.f,0.f,0.f);
    pushParticles(sphere, velo, color);
}

//----------------------------------------------------------------------
int FLOCK::addHose(int total_n, float4 center, float4 velocity, float radius, float4 color)
{
    radius *= spacing;
    Hose* hose = new Hose(ps, total_n, center, velocity, radius, spacing, color);
    hoses.push_back(hose);
    return hoses.size()-1;

}

//----------------------------------------------------------------------
void FLOCK::updateHose(int index, float4 center, float4 velocity, float radius, float4 color)
{
    //we need to expose the vector of hoses somehow doesn't seem right to make user manage an index
    radius *= spacing;
    hoses[index]->update(center, velocity, radius, spacing, color);
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
    std::vector<float4> poss(40);
    
    float4 posx(100.,100.,100.,1.);
    std::fill(poss.begin(), poss.end(),posx);
    
    cl_position_u.acquire();
    cl_position_u.copyToDevice(poss);
    cl_position_u.release();
    ps->cli->queue.finish();
}

//----------------------------------------------------------------------
void FLOCK::pushParticles(vector<float4> pos, float4 velo, float4 color)
{
    int nn = pos.size();
    std::vector<float4> vels(nn);
    float ms = flock_params.max_speed;
    
#if 1 
    std::fill(vels.begin(), vels.end(), velo);
#endif    

#if 0 
    for(int i=0; i < nn; i++){
        vels[i] = float4(rand(), rand(), rand(), velo.w);
        vels[i] = normalize3(vels[i]);
        vels[i] = vels[i] *  ms;
    }
#endif

    pushParticles(pos, vels, color);
}

//----------------------------------------------------------------------
void FLOCK::pushParticles(vector<float4> pos, vector<float4> vels, float4 color)
{
    int nn = pos.size();
    
    // if we have reach max num of particles, then return
    if (num + nn > max_num) {return;}
    
    std::vector<float4> cols(nn);
    std::fill(cols.begin(), cols.end(),color); //BLENDER

#ifdef CPU
    std::copy(pos.begin(), pos.end(), positions.begin()+num);
#endif

#ifdef GPU
    glFinish();
    cl_position_u.acquire();
    cl_color_u.acquire();
 
    cl_position_u.copyToDevice(pos, num);
    cl_color_u.copyToDevice(cols, num);
    cl_velocity_u.copyToDevice(vels, num);

    settings->SetSetting("Number of Particles", num+nn);
    updateFLOCKP();

    num += nn;  //keep track of number of particles we use
    
    cl_color_u.release();
    cl_position_u.release();

#else
    num += nn;  //keep track of number of particles we use
#endif

	renderer->setNum(num);
}

//----------------------------------------------------------------------
void FLOCK::render()
{
    renderer->render_box(grid->getBndMin(), grid->getBndMax());
    System::render();
}

//----------------------------------------------------------------------
void FLOCK::setRenderer()
{
    switch(ps->settings->getRenderType())
    {
        case RTPSettings::SPRITE_RENDER:
            renderer = new SpriteRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
			printf("new SpriteRender\n");
            break;
        case RTPSettings::SCREEN_SPACE_RENDER:
			printf("new SSFRender\n");
            renderer = new SSFRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        case RTPSettings::RENDER:
			printf("new Render\n");
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        case RTPSettings::SPHERE3D_RENDER:
			printf("new Sphere3DRender\n");
            renderer = new Sphere3DRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
            break;
        default:
			printf("new Render in default\n");
            renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
        break;
    }
    renderer->setParticleRadius(spacing);
}


} 
