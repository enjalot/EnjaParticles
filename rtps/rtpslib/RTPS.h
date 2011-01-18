#ifndef RTPS_RTPS_H_INCLUDED
#define RTPS_RTPS_H_INCLUDED

#include <vector>

//defines useful structs like float3 and float4
#include "util.h"



//OpenCL API
#include "opencl/CLL.h"

//Render API
#include "render/Render.h"

//System API
#include "system/System.h"

//initial value API
#include "domain/IV.h"

//settings class to configure the framework
#include "RTPSettings.h"

namespace rtps {

class RTPS
{
public:
    //default constructor
    RTPS();
    //Setup CL, Render, initial values and System based on settings
    RTPS(RTPSettings s);

    ~RTPS();

    void Init();

    //Keep track of settings
    RTPSettings settings;
    
    //OpenCL abstraction instance
    CL *cli;
    Render *renderer;

    //will be instanciated as a specific subclass like SPH or Boids
    System *system;
    //std::vector<System> systems;

    //initial value helper
    //IV iv;

    void update();
    void render();
    
    //should this be private?
    void updateNum(int num);
    
    //temporary render control
    bool render_particles;
    bool render_ghosts;
    void printTimers();


};

}

#endif
