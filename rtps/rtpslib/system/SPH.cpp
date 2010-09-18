
#include <GL/glew.h>
#include <math.h>

#include "SPH.h"
#include "../particle/UniformGrid.h"

namespace rtps {


SPH::SPH(RTPS *psfr, int n)
{
    //store the particle system framework
    ps = psfr;

    num = n;

    //*** Initialization, TODO: move out of here to the particle directory
    std::vector<float4> positions(num);
    std::vector<float4> colors(num);
    std::vector<float4> forces(num);
    std::vector<float4> velocities(num);
    std::vector<float> densities(num);
    
    
    //std::fill(positions.begin(), positions.end(),(float4) {0.0f, 0.0f, 0.0f, 1.0f});
    //init sph stuff
    sph_settings.rest_density = 1000;
    sph_settings.simulation_scale = .001;

    sph_settings.particle_mass = (128*1024)/num * .0002;
    printf("particle mass: %f\n", sph_settings.particle_mass);
    sph_settings.particle_rest_distance = .87 * pow(sph_settings.particle_mass / sph_settings.rest_density, 1./3.);
    printf("particle rest distance: %f\n", sph_settings.particle_rest_distance);
   
    sph_settings.smoothing_distance = 2.f * sph_settings.particle_rest_distance;
    sph_settings.boundary_distance = .5f * sph_settings.particle_rest_distance;

    sph_settings.spacing = sph_settings.particle_rest_distance / sph_settings.simulation_scale;
    float particle_radius = sph_settings.spacing;
    printf("particle radius: %f\n", particle_radius);

    //grid = UniformGrid(float3(0,0,0), float3(1024, 1024, 1024), sph_settings.smoothing_distance / sph_settings.simulation_scale);
    grid = UniformGrid(float3(-512,0,-512), float3(512, 1024, 512), sph_settings.smoothing_distance / sph_settings.simulation_scale);
    grid.make_cube(&positions[0], sph_settings.spacing, num);


/*
typedef struct SPHParams
{
    float4 grid_min;
    float4 grid_max;
    float mass;
    float rest_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
 
} SPHParams;
*/

    params.grid_min = grid.getMin();
    params.grid_max = grid.getMax();
    params.mass = sph_settings.particle_mass;
    params.rest_distance = sph_settings.particle_rest_distance;
    params.smoothing_distance = sph_settings.smoothing_distance;
    params.simulation_scale = sph_settings.simulation_scale;
    params.boundary_stiffness = 10000.0f;
    params.boundary_dampening = 256.0f;
    params.boundary_distance = sph_settings.particle_rest_distance * .5f;
    params.EPSILON = .00001f;
    params.PI = 3.14159265f;
    params.K = 1.5f;
 
    //TODO make a helper constructor for buffer to make a cl_mem from a struct
    std::vector<SPHParams> vparams(0);
    vparams.push_back(params);
    cl_params = Buffer<SPHParams>(ps->cli, vparams);


    std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.0f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

    std::fill(densities.begin(), densities.end(), 0.0f);

    /*
    for(int i = 0; i < 20; i++)
    {
        printf("position[%d] = %f %f %f\n", positions[i].x, positions[i].y, positions[i].z);
    }
    */

    //*** end Initialization
   



    // VBO creation, TODO: should be abstracted to another class
    managed = true;
    printf("positions: %d, %d, %d\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);
    // end VBO creation

    //vbo buffers
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);

    //pure opencl buffers
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);

    cl_density = Buffer<float>(ps->cli, densities);



    printf("create density kernel\n");
    loadDensity();

    printf("create pressure kernel\n");
    loadPressure();

    printf("create viscosity kernel\n");
    loadViscosity();


    printf("create collision wall kernel\n");
    loadCollision_wall();

    //could generalize this to other integration methods later (leap frog, RK4)
    printf("create euler kernel\n");
    loadEuler();
    printf("euler kernel created\n");


    //check the params
    //std::vector<SPHParams> test = cl_params.copyToHost(1);
    //printf("mass: %f, EPSILON %f \n", test[0].mass, test[0].EPSILON);    


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
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
    
    //printf("execute!\n");

    for(int i=0; i < 10; i++)
    {
        k_density.execute(num);
        k_pressure.execute(num);
        k_viscosity.execute(num);

        //k_collision_wall.execute(num);

        //euler integration
        k_euler.execute(num);
    }

    std::vector<float4> ftest = cl_force.copyToHost(100);
    for(int i = 0; i < 100; i++)
    {
        if(ftest[i].z != 0.0)
            printf("force: %f %f %f  \n", ftest[i].x, ftest[i].y, ftest[i].z);
    }
    printf("execute!\n");


    cl_position.release();
    cl_color.release();
}

}
