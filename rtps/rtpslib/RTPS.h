#ifndef RTPS_RTPS_H_INCLUDED
#define RTPS_RTPS_H_INCLUDED

#include <vector>

//OpenCL API
#include "opencl/CL.h"

//Render API
#include "render/Render.h"

//System API
#include "system/System.h"

//initial value API
#include "particle/IV.h"

//settings class to configure the framework
#include "RPTSettings.h"


namespace rtps {

class RTPS
{
public:
    RTPS();

    Init();

    //Keep track of settings
    RTPSettings settings;
    
    //OpenCL abstraction instance
    CL cli;
    Render render;

    //will be instanciated as a specific subclass like SPH or Boids
    System system;
    //std::vector<System> systems;

    //initial value helper
    IV iv;
};

}

#endif
