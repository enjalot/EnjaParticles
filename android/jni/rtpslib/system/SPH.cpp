
#include <math.h>
using namespace std;

#include "System.h"
#include "SPH.h"
#include "../domain/Domain.h"
#include "../domain/IV.h"


#include <android/log.h>

#include <time.h>
namespace rtps {


SPH::SPH(RTPS *psfr, int n)
{
    //store the particle system framework
    ps = psfr;

    num = 0;
    max_num = n;

    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);
    veleval.resize(max_num);
    densities.resize(max_num);
    xsphs.resize(max_num);

    srand( time(NULL) );
        
    //std::fill(positions.begin(), positions.end(),(float4) {0.0f, 0.0f, 0.0f, 1.0f});
    //init sph stuff
    sph_settings.simulation_scale = .08;

    grid = ps->settings.grid;
    calculateSPHSettings();
    setupDomain();
    sph_settings.integrator = LEAPFROG;

    prepareSorted();

    

    /*
    for(int i = 0; i < 20; i++)
    {
        printf("position[%d] = %f %f %f\n", positions[i].x, positions[i].y, positions[i].z);
    }
    */

    //*** end Initialization

}

SPH::~SPH()
{
    if(pos_vbo && managed)
    {
        glBindBuffer(1, pos_vbo);
        glDeleteBuffers(1, (GLuint*)&pos_vbo);
        pos_vbo = 0;
    }
    if(col_vbo && managed)
    {
        glBindBuffer(1, col_vbo);
        glDeleteBuffers(1, (GLuint*)&col_vbo);
        col_vbo = 0;
    }
}

void SPH::update()
{
    //call kernels
    //TODO: add timings

    __android_log_print(ANDROID_LOG_INFO, "RTPS", "update!");
    cpuDensity();
    cpuPressure();
    cpuViscosity();
    cpuXSPH();
    cpuCollision_wall();

    //cpuEuler();
    cpuLeapFrog();
    //printf("positions[0].z %f\n", positions[0].z);
    /*
    for(int i = 0; i < 100; i++)
    {
 //       if(xsphs[i].z != 0.0)
            //printf("force: %f %f %f  \n", veleval[i].x, veleval[i].y, veleval[i].z);
            printf("force: %f %f %f  \n", xsphs[i].x, xsphs[i].y, xsphs[i].z);
            //printf("force: %f %f %f  \n", velocities[i].x, velocities[i].y, velocities[i].z);
    }
    */

    __android_log_print(ANDROID_LOG_INFO, "RTPS", "update: push vbo");

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_num* sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_num * sizeof(float4), &colors[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

    __android_log_print(ANDROID_LOG_INFO, "RTPS", "update: done!");




}

void SPH::calculateSPHSettings()
{
    /*!
    * The Particle Mass (and hence everything following) depends on the MAXIMUM number of particles in the system    */
    sph_settings.rest_density = 1000;

    sph_settings.particle_mass = (128*1024.0)/max_num * .0002;
    //printf("particle mass: %f\n", sph_settings.particle_mass);
    //constant .87 is magic
    sph_settings.particle_rest_distance = .87 * pow(sph_settings.particle_mass / sph_settings.rest_density, 1./3.); 
    //printf("particle rest distance: %f\n", sph_settings.particle_rest_distance);
    
    //messing with smoothing distance, making it really small to remove interaction still results in weird force values
    sph_settings.smoothing_distance = 2.0f * sph_settings.particle_rest_distance;
    sph_settings.boundary_distance = .5f * sph_settings.particle_rest_distance;

    sph_settings.spacing = sph_settings.particle_rest_distance/ sph_settings.simulation_scale;

    float particle_radius = sph_settings.spacing;
    //printf("particle radius: %f\n", particle_radius);
 
    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.simulation_scale = sph_settings.simulation_scale;
    params.boundary_stiffness = 20000.0f;
    params.boundary_dampening = 256.0f;
    params.boundary_distance = sph_settings.particle_rest_distance * .5f;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 20.0f;
    params.num = num;

    float h = params.smoothing_distance;
    float pi = acos(-1.0);
    float h9 = pow(h,9.);
    float h6 = pow(h,6.);
    float h3 = pow(h,3.);
    params.wpoly6_coef = 315.f/64.0f/pi/h9;
    params.wpoly6_d_coef = -945.f/(32.0f*pi*h9);
    params.wpoly6_dd_coef = -945.f/(32.0f*pi*h9);
    params.wspiky_coef = 15.f/pi/h6;
    params.wspiky_d_coef = 45.f/(pi*h6);
    params.wvisc_coef = 15./(2.*pi*h3);
    params.wvisc_d_coef = 15./(2.*pi*h3);
    params.wvisc_dd_coef = 45./(pi*h6);
}

void SPH::prepareSorted()
{

    std::fill(positions.begin(), positions.end(),float4(1.0f, 1.0f, 1.0f, 0.0f));
    std::fill(colors.begin(), colors.end(),float4(0.0f, 1.0f, 0.0f, 0.0f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);
    std::fill(xsphs.begin(), xsphs.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    // VBO creation, TODO: should be abstracted to another class
    managed = true;
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "init positions.size(): %d", positions.size());

    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pos_vbo: %d", pos_vbo);
    //printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "col_vbo: %d", col_vbo);
    //printf("col vbo: %d\n", col_vbo);
    // end VBO creation
}


void SPH::setupDomain()
{
    grid.calculateCells(sph_settings.smoothing_distance / sph_settings.simulation_scale);


    grid_params.grid_min = grid.getMin();
    grid_params.grid_max = grid.getMax();
    grid_params.bnd_min  = grid.getBndMin();
    grid_params.bnd_max  = grid.getBndMax();
    grid_params.grid_res = grid.getRes();
    grid_params.grid_size = grid.getSize();
    grid_params.grid_delta = grid.getDelta();
    grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

    //printf("gp nb_cells: %d\n", grid_params.nb_cells);


    /*
    grid_params.grid_inv_delta.x = 1. / grid_params.grid_delta.x;
    grid_params.grid_inv_delta.y = 1. / grid_params.grid_delta.y;
    grid_params.grid_inv_delta.z = 1. / grid_params.grid_delta.z;
    grid_params.grid_inv_delta.w = 1.;
    */

    float ss = sph_settings.simulation_scale;

    grid_params_scaled.grid_min = grid_params.grid_min * ss;
    grid_params_scaled.grid_max = grid_params.grid_max * ss;
    grid_params_scaled.bnd_min  = grid_params.bnd_min * ss;
    grid_params_scaled.bnd_max  = grid_params.bnd_max * ss;
    grid_params_scaled.grid_res = grid_params.grid_res;
    grid_params_scaled.grid_size = grid_params.grid_size * ss;
    grid_params_scaled.grid_delta = grid_params.grid_delta / ss;
    //grid_params_scaled.nb_cells = (int) (grid_params_scaled.grid_res.x*grid_params_scaled.grid_res.y*grid_params_scaled.grid_res.z);
    grid_params_scaled.nb_cells = grid_params.nb_cells;
    //grid_params_scaled.grid_inv_delta = grid_params.grid_inv_delta / ss;
    //grid_params_scaled.grid_inv_delta.w = 1.0f;

    grid_params.print();
    grid_params_scaled.print();

}

int SPH::addBox(int nn, float4 min, float4 max, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = sph_settings.simulation_scale;
    }
    vector<float4> rect = addRect(nn, min, max, sph_settings.spacing, scale);
    pushParticles(rect);
    return rect.size();
}

void SPH::addBall(int nn, float4 center, float radius, bool scaled)
{
    float scale = 1.0f;
    if(scaled)
    {
        scale = sph_settings.simulation_scale;
    }
    vector<float4> sphere = addSphere(nn, center, radius, sph_settings.spacing, scale);
    pushParticles(sphere);
}

void SPH::pushParticles(vector<float4> pos)
{
    int nn = pos.size();
    if (num + nn > max_num) {return;}
    float rr = (rand() % 255)/255.0f;
    float4 color(rr, 1.0f, 1.0f - rr, 1.0f);
    //printf("random: %f\n", rr);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp random: %f", rr);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp nn: %d num: %d", nn, num);

    std::vector<float4> cols(nn);
    std::vector<float4> vels(nn);

    std::fill(cols.begin(), cols.end(),color);
    float v = .0f;
    float4 iv = float4(v, v, -v, 0.0f);
    std::fill(velocities.begin()+num, velocities.begin()+num+nn,iv);

    std::copy(pos.begin(), pos.end(), positions.begin()+num); 
    std::copy(cols.begin(), cols.end(), colors.begin()+num); 
    int i = 0;
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp positions[%d]: %f, %f, %f", i, positions[i].x, positions[i].y, positions[i].z);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp colors[%d]: %f, %f, %f", i, colors[i].x, colors[i].y, colors[i].z);
    i = 1;
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp positions[%d]: %f, %f, %f", i, positions[i].x, positions[i].y, positions[i].z);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp colors[%d]: %f, %f, %f", i, colors[i].x, colors[i].y, colors[i].z);

    __android_log_print(ANDROID_LOG_INFO, "RTPS", "pp positions.size(): %d", positions.size());

    num += nn;
    ps->updateNum(num);
    //need to push to VBOs still
}


} //end namespace
