#include <stdio.h>

#include <GL/glew.h>

#include "System.h"
#include "Simple.h"

namespace rtps {


Simple::Simple(RTPS *psfr, int n)
{
    max_num = n;
    num = max_num;
    //store the particle system framework
    ps = psfr;
    grid = ps->settings.grid;

    printf("num: %d\n", num);
    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);

    float4 min = grid.getBndMin();
    float4 max = grid.getBndMax();

    float spacing = .1; 
    std::vector<float4> box = addRect(num, min, max, spacing, 1);     std::copy(box.begin(), box.end(), positions.begin());


    //std::fill(positions.begin(), positions.end(), float4(0.0f, 0.0f, 0.0f, 1.0f));
    std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.0f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    
    managed = true;
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);

#if GPU
    //vbo buffers
    printf("making cl_buffers\n");
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);
    printf("done with cl_buffers\n");



    //pure opencl buffers
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);;

    //could generalize this to other integration methods later (leap frog, RK4)
    printf("create euler kernel\n");
    loadEuler();
#endif

   }

Simple::~Simple()
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

void Simple::update()
{
#ifdef CPU

    //printf("calling cpuEuler\n");
    cpuEuler();

    //printf("pushing positions to gpu\n");
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float4), &colors[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

    //printf("done pushing to gpu\n");


#endif
#ifdef GPU

    //call kernels
    //add timings
    glFinish();
    //err = queue.enqueueAcquireGLObjects(&enjas->cl_vbos, NULL, &event);
    cl_position.acquire();
    cl_color.acquire();
    //printf("acquire: %s\n", oclErrorString(err));
    
    //euler integration
    //err = queue.enqueueNDRangeKernel(k_euler, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    //queue.finish();
    //printf("executing euler in simple!\n");
    k_euler.execute(num);

    //err = queue.enqueueReleaseGLObjects(&enjas->cl_vbos, NULL, &event);
    cl_position.release();
    cl_color.release();
    //printf("release gl: %s\n", oclErrorString(err));
#endif
}


}
