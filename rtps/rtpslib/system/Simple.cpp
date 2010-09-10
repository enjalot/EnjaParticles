#include <stdio.h>

#include <GL/glew.h>
#include "Simple.h"

namespace rtps {


Simple::Simple(RTPS *psfr, int n)
{
    num = n;
    //store the particle system framework
    ps = psfr;

    //we take some variables that EnjaParticles have already prepared
    //
    //create buffers
    //cl_force = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Vec4)*num, NULL, &err); //float4 array

    //initialize force
    // don't need to because it gets set to 0 first in pressure calculation
    //std::vector<float> dens(num);
    //std::vector<Vec4> forc(num);
    //std::fill(dens.begin(), dens.end(), 0.0f);
    //Vec4 f(0.0f, 0.0f, -9.8f, 0.0f);
    //std::fill(forc.begin(), forc.end(), f);
    //err = queue.enqueueWriteBuffer(cl_force, CL_TRUE, 0, sizeof(Vec4)*num, &forc[0], NULL, &event);

    //cl_velocity = cl::Buffer(context, CL_MEM_READ_WRITE, 4*sizeof(float)*num, NULL, &err); //float4 array

    std::vector<float4> positions(num);
    std::vector<float4> colors(num);
    std::vector<float4> forces(num);
    std::vector<float4> velocities(num);
    
    
    std::fill(positions.begin(), positions.end(),(float4) {0.0f, 0.0f, 0.0f, 1.0f});
    std::fill(colors.begin(), colors.end(),(float4) {1.0f, 0.0f, 0.0f, 0.0f});
    std::fill(forces.begin(), forces.end(),(float4) {0.0f, 0.0f, 1.0f, 0.0f});
    std::fill(velocities.begin(), velocities.end(),(float4) {0.0f, 0.0f, -9.8f, 0.0f});

    
    managed = true;
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);

    //vbo buffers
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);

    //pure opencl buffers
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);;

    //could generalize this to other integration methods later (leap frog, RK4)
    printf("create euler kernel\n");
    
    #include "simple/euler.cl"
    printf("%s\n", euler_program_source.c_str());
    k_euler = Kernel(ps->cli, euler_program_source, "euler");
  
    //TODO: fix the way we are wrapping buffers
    k_euler.setArg(0, cl_position.cl_buffer[0]);
    k_euler.setArg(1, cl_velocity.cl_buffer[0]);
    k_euler.setArg(2, cl_force.cl_buffer[0]);
    k_euler.setArg(3, .01f); //time step (should be set from settings)
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
    k_euler.execute(num);

    //err = queue.enqueueReleaseGLObjects(&enjas->cl_vbos, NULL, &event);
    cl_position.release();
    cl_color.release();
    //printf("release gl: %s\n", oclErrorString(err));

}


}
