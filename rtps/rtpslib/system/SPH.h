#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
//#include "../util.h"


namespace rtps {

class SPH : public System
{
public:
    SPH(RTPS *ps, int num);
    ~SPH();

    void update();

    //the particle system framework
    RTPS *ps;
    int num;


    Kernel k_density, k_pressure, k_viscosity;
    Kernel k_collision_wall;
    Kernel k_euler;

    Buffer<float4> cl_position;
    Buffer<float4> cl_color;
    Buffer<float> cl_density;
    Buffer<float4> cl_force;
    Buffer<float4> cl_velocity;

    //these are defined in sph/ folder next to the kernels
    Kernel loadDensity();
    Kernel loadPressure();
    Kernel loadViscosity();
    Kernel loadCollision_wall();
    Kernel loadEuler();

};

}

#endif
