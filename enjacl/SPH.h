#ifndef ENJA_SPH_H_INCLUDED
#define ENJA_SPH_H_INCLUDED



#include <string>

#include "enja.h"
#include "system.h"
#include <CL/cl.hpp>

class SPH : public System
{
public:
    SPH(EnjaParticles *enjas);
    ~SPH();

    void update();

    EnjaParticles *enjas;
    int num; //number of particles (taken from enjas)

    cl::Context context;
    cl::CommandQueue queue;
    int err;
    cl::Event event;

    //cl source codes
    static const std::string sources[];
    enum {DENSITY, PRESSURE, VISCOSITY, COLLISION_WALL, EULER};

    cl::Kernel k_density, k_pressure, k_viscosity;
    cl::Kernel k_collision_wall;
    cl::Kernel k_euler;

    //the vbos will come from enjas
    //std::vector<cl::Memory> cl_vbos;  //0: position vbo, 1: color vbo
    cl::Buffer cl_density;
    cl::Buffer cl_force;
    cl::Buffer cl_velocity;

};

#endif
