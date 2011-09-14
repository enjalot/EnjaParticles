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


#ifndef RTPS_FLOCK_H_INCLUDED
#define RTPS_FLOCK_H_INCLUDED

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <string>

#include <RTPS.h>
#include <System.h>
#include <Kernel.h>
#include <Buffer.h>

#include <Domain.h>
#include <FLOCKSettings.h>

#include <Hash.h>
#include <BitonicSort.h>
#include <CellIndices.h>
#include <Permute.h>

#include <flock/Rules.h>
#include <flock/EulerIntegration.h>

#include <Hose.h>

#include <timer_eb.h>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{

class RTPS_EXPORT FLOCK : public System
{


public:
    FLOCK(RTPS *ps, int num);
    ~FLOCK();

    // update call for CPU and GPU
    void update();
    
    //wrapper around IV.h addRect
    int addBox(int nn, float4 min, float4 max, bool scaled, float4 color=float4(1., 0., 0., 1.));
    
    //wrapper around IV.h addSphere
    void addBall(int nn, float4 center, float radius, bool scaled, float4 color=float4(1., 0., 0., 1.));

    //wrapper around Hose.h 
    int addHose(int total_n, float4 center, float4 velocity, float radius, float4 color=float4(1.0, 0.0, 0.0, 1.0f));
    void updateHose(int index, float4 center, float4 velocity, float radius, float4 color=float4(1.0, 0.0, 0.0, 1.0f));
    void sprayHoses();

    virtual void render();
    
    void testDelete();
    
    // timers
    EB::TimerList timers;
    int setupTimers();
    void printTimers();

    void pushParticles(vector<float4> pos, float4 velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));
    void pushParticles(vector<float4> pos, vector<float4> velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));

protected:
    virtual void setRenderer();
    
private:
    //the particle system framework
    RTPS *ps;
    RTPSettings *settings;

    FLOCKParameters flock_params;
    GridParams      grid_params;
    GridParams      grid_params_scaled;
    float spacing; //Particle rest distance in world coordinates

    std::string flock_source_dir;

    int nb_var;

    std::vector<float4> deleted_pos;
    std::vector<float4> deleted_vel;
    
    //keep track of hoses
    std::vector<Hose*> hoses;   
    
    //needs to be called when particles are added
    void calculateFLOCKSettings();
    void setupDomain();
    void prepareSorted();
    
    //This should be in OpenCL classes
    Kernel k_scopy;

    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float4> velocities;
    std::vector<float4> veleval;
    
    std::vector<int4>   flockmates; //x will store the num of flockmates and y will store the num of flockmates within the min dist (for separation rule)
    std::vector<float4> separation;
    std::vector<float4> alignment;
    std::vector<float4> cohesion;
    std::vector<float4> goal;
    std::vector<float4> avoid;
    std::vector<float4> wander;
    std::vector<float4> leaderfollowing;

    Buffer<float4>      cl_position_u;
    Buffer<float4>      cl_position_s;
    Buffer<float4>      cl_color_u;
    Buffer<float4>      cl_color_s;
    Buffer<float4>      cl_velocity_u;
    Buffer<float4>      cl_velocity_s;
    Buffer<float4>      cl_veleval_u;
    Buffer<float4>      cl_veleval_s;
    
    Buffer<int4>        cl_flockmates_s;
    Buffer<float4>      cl_separation_s;
    Buffer<float4>      cl_alignment_s;
    Buffer<float4>      cl_cohesion_s;
    Buffer<float4>      cl_goal_s;
    Buffer<float4>      cl_avoid_s;
    Buffer<float4>      cl_leaderfollowing_s;

    //Neighbor Search related arrays
	Buffer<unsigned int> 		cl_cell_indices_start;
	Buffer<unsigned int> 		cl_cell_indices_end;
	Buffer<unsigned int> 		cl_sort_hashes;
	Buffer<unsigned int> 		cl_sort_indices;
	
    //Two arrays for bitonic sort (sort not done in place)
	Buffer<unsigned int>         cl_sort_output_hashes;
	Buffer<unsigned int>         cl_sort_output_indices;

    Bitonic<unsigned int>        bitonic;
    
    //Parameter structs
    Buffer<FLOCKParameters>     cl_FLOCKParameters;
	Buffer<GridParams>          cl_GridParams;
	Buffer<GridParams>          cl_GridParamsScaled;
   
    Buffer<float4>  	clf_debug;  //just for debugging cl files
	Buffer<int4>		cli_debug;  //just for debugging cl files
    
    //CPU functions
    void cpuComputeRules();
    void cpuAverageRules();
	void cpuRules();
	void cpuEulerIntegration();

    void updateCPU();
    void updateGPU();

    // calculate the various parameters that depend on max_num of particles
    void calculate();

    //copy the FLOCK  parameter struct to the GPU
    void updateFLOCKP();

    //Nearest Neighbors search related functions
    void call_prep(int stage);
    Hash hash;
    CellIndices cellindices;
    Permute permute;
    void hash_and_sort();
    void bitonic_sort();
    
    Rules rules;
    EulerIntegration euler_integration;
    
    void integrate();
   
};

}

#endif
