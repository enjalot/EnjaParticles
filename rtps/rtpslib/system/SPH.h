#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
//#include "../util.h"
#include "../particle/UniformGrid.h"


namespace rtps {

//keep track of the fluid settings
typedef struct SPHSettings
{
    float rest_density;
    float simulation_scale;
    float particle_mass;
    float particle_rest_distance;
    float smoothing_distance;
    float boundary_distance;
    float spacing;
    float grid_cell_size;

} SPHSettings;

//pass parameters to OpenCL routines
typedef struct SPHParams
{
    float3 grid_min;
    //float grid_min_padding;     //float3s take up a float4 of space in OpenCL 1.0 and 1.1
    float3 grid_max;
    //float grid_max_padding;
//    int num;
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //gas constant
 
} SPHParams __attribute__((aligned(16)));

class SPH : public System
{
public:
    SPH(RTPS *ps, int num);
    ~SPH();

    void update();

private:
    //the particle system framework
    RTPS *ps;

    SPHSettings sph_settings;
    SPHParams params;

    Kernel k_density, k_pressure, k_viscosity;
    Kernel k_collision_wall;
    Kernel k_euler;

    Buffer<SPHParams> cl_params;


    std::vector<float4> positions;
    std::vector<float> densities;
    std::vector<float4> forces;
    std::vector<float4> velocities;

    Buffer<float4> cl_position;
    Buffer<float4> cl_color;
    Buffer<float> cl_density;
    Buffer<float4> cl_force;
    Buffer<float4> cl_velocity;
    
    Buffer<float4> cl_error_check;

    //these are defined in sph/ folder next to the kernels
    void loadDensity();
    void loadPressure();
    void loadViscosity();
    void loadXSPH();
    void loadCollision_wall();
    void loadEuler();

    //CPU functions
    void cpuDensity();
    void cpuPressure();
    void cpuViscosity();
    void cpuXSPH();
    void cpuEuler();
    void cpuCollision_wall();

    float Wpoly6(float4 r, float h);
    float Wspiky(float4 r, float h);
    float Wviscosity(float4 r, float h);

};



}

#endif
