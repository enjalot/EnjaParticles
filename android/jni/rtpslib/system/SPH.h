#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>
#include <stdlib.h>

#include "../RTPS.h"
#include "System.h"
//#include "../util.h"
#include "../domain/Domain.h"


namespace rtps {

enum Integrator {EULER, LEAPFROG};

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
    Integrator integrator;

} SPHSettings;

//pass parameters to OpenCL routines
typedef struct SPHParams
{
    float4 grid_min;
    //float grid_min_padding;     //float3s take up a float4 of space in OpenCL 1.0 and 1.1
    float4 grid_max;
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
 
    float friction_coef;
    float restitution_coef;
    float shear;
    float attraction;
    float spring;
    float gravity; // -9.8 m/sec^2

    //Kernel Coefficients
    float wpoly6_coef;
    float wpoly6_d_coef;
    float wpoly6_dd_coef; // laplacian
    float wspiky_coef;
    float wspiky_d_coef;
    float wspiky_dd_coef;
    float wvisc_coef;
    float wvisc_d_coef;
    float wvisc_dd_coef;

    int num;
    int nb_vars; // for combined variables (vars_sorted, etc.)
    int choice; // which kind of calculation to invoke


    void print() {
        printf("----- SPHParams ----\n");
        printf("simulation_scale: %f\n", simulation_scale);
        printf("friction_coef: %f\n", friction_coef);
        printf("restitution_coef: %f\n", restitution_coef);
        printf("damping: %f\n", boundary_dampening);
        printf("shear: %f\n", shear);
        printf("attraction: %f\n", attraction);
        printf("spring: %f\n", spring);
        printf("gravity: %f\n", gravity);
        printf("choice: %d\n", choice);
    }


} SPHParams __attribute__((aligned(16)));

class SPH : public System
{
public:
    SPH(RTPS *ps, int num);
    ~SPH();

    void update();

    //wrapper around IV.h addRect
    int addBox(int nn, float4 min, float4 max, bool scaled);
    //wrapper around IV.h addSphere
    void addBall(int nn, float4 center, float radius, bool scaled);


private:
    //the particle system framework
    RTPS *ps;

    SPHSettings sph_settings;
    SPHParams params;
    GridParams grid_params;
    GridParams grid_params_scaled;

    //needs to be called when particles are added
    void calculateSPHSettings();
    void setupDomain();
    void prepareSorted();
    void pushParticles(std::vector<float4> pos);


    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float> densities;
    std::vector<float4> forces;
    std::vector<float4> velocities;
    std::vector<float4> veleval;
    std::vector<float4> xsphs;

    //CPU functions
    void cpuDensity();
    void cpuPressure();
    void cpuViscosity();
    void cpuXSPH();
    void cpuCollision_wall();
    void cpuEuler();
    void cpuLeapFrog();


    float Wpoly6(float4 r, float h);
    float Wspiky(float4 r, float h);
    float Wviscosity(float4 r, float h);

};



}

#endif
