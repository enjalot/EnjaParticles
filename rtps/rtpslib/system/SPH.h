#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>

#include "RTPS.h"
#include "system.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"

namespace rtps {

class SPH : public System
{
public:
    SPH(RTPS *ps, num);
    ~SPH();

    void update();

    //the particle system framework
    RTPS *ps;
    //number of particles
    int num; 

    /*
    cl::Context context;
    cl::CommandQueue queue;
    int err;
    cl::Event event;
    */
    //cl source codes
    /*
    static const std::string sources[];
    enum {DENSITY, PRESSURE, VISCOSITY, COLLISION_WALL, EULER};
    */

    Kernel k_density, k_pressure, k_viscosity;
    Kernel k_collision_wall;
    Kernel k_euler;

    //the vbos will come from enjas
    //std::vector<cl::Memory> cl_vbos;  //0: position vbo, 1: color vbo
    Buffer cl_position;
    Buffer cl_color;
    Buffer cl_density;
    Buffer cl_force;
    Buffer cl_velocity;

};

}

#endif
