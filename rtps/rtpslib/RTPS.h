#ifndef RTPS_RTPS_H_INCLUDED
#define RTPS_RTPS_H_INCLUDED

#include <vector>

//System API
#include "system/System.h"

//OpenCL API
#include "opencl/CLL.h"

//initial value API
//TODO probably shouldn't be included here
#include "domain/IV.h"

//settings class to configure the framework
#include "RTPSettings.h"

//defines useful structs like float3 and float4
#include "structs.h"

//defines a few handy utility functions
//TODO should not be included here
#include "util.h"

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

    class RTPS_EXPORT RTPS
    
    {
    public:
        //default constructor
        RTPS();
        //Setup CL, Render, initial values and System based on settings
        RTPS(RTPSettings s);
        RTPS(RTPSettings s, CL* _cli);

        ~RTPS();

        void Init();

        //Keep track of settings
        RTPSettings settings;
        
        //OpenCL abstraction instance
        //TODO shouldn't be public
        CL *cli;
        //Render *renderer;

        //will be instanciated as a specific subclass like SPH or Boids
        //TODO shouldn't be public? right now we expose various methods from the system
        System *system;
        //std::vector<System> systems;

        //initial value helper
        //IV iv;

        void update();
        void render();

        void printTimers();

    private:
        bool cl_managed;
        
    };
}

#endif
