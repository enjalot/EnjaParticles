#ifndef RTPS_RTPS_H_INCLUDED
#define RTPS_RTPS_H_INCLUDED

#include <vector>

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

//defines useful structs like float3 and float4
#include "util.h"

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

    RTPSettings getRTPSettings() { return settings; }
    //std::vector<System> systems;

    //initial value helper
    //IV iv;

    // setters for point size and transformation
    void SetPointScale(float p) { renderer->pointscale = p; }

     void SetTransformation(float4 t[4]) {
         for(int i=0; i < 4; i++)
             system->transformation[i] = t[i];
    }

    void update();
    void render();
    
    //should this be private?
    void updateNum(int num);
    
    void printTimers();


};

}

#endif
