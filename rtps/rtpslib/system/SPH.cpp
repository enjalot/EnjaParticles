
#include <GL/glew.h>
#include "SPH.h"

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
    for(int i = 0; i < num; i++)
    {
        positions[i] = float4((i*1.0f)/num, 0.0f, 0.0f, 1.0f);
    }
    std::fill(colors.begin(), colors.end(),(float4) {1.0f, 0.0f, 0.0f, 0.0f});
    std::fill(forces.begin(), forces.end(),(float4) {0.0f, 0.0f, 1.0f, 0.0f});
    std::fill(velocities.begin(), velocities.end(),(float4) {0.0f, 0.0f, -9.8f, 0.0f});

    std::fill(densities.begin(), densities.end(), 0.0f);
    //*** end Initialization
   


    // VBO creation, TODO: should be abstracted to another class
    managed = true;
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


    //could generalize this to other integration methods later (leap frog, RK4)
    printf("create euler kernel\n");
    k_euler = loadEuler();
    printf("euler kernel created\n");



  


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
    
    printf("execute!\n");
    //euler integration
    //k_euler.execute(num);

    cl_position.release();
    cl_color.release();

}


}
