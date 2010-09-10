#include <stdio.h>

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


    printf("create euler kernel\n");
    #include "simple/euler.cl"
    //could generalize this to other integration methods later (leap frog, RK4)
    k_euler = new Kernel(ps->cli, euler_program_source, "euler");
    k_euler->setArg(0, *cl_position);
    k_euler->setArg(1, *cl_velocity);
    k_euler->setArg(2, *cl_force);
    k_euler->setArg(3, .01f); //time step (should be set from settings)
}

Simple::~Simple()
{
    delete k_euler;
    delete cl_position;
    delete cl_color;
    delete cl_force;
    delete cl_velocity;
}

void Simple::update()
{
    //call kernels
    //add timings
    glFinish();
    //err = queue.enqueueAcquireGLObjects(&enjas->cl_vbos, NULL, &event);
    cl_position->acquire();
    cl_color->acquire();
    //printf("acquire: %s\n", oclErrorString(err));
    
    //euler integration
    //err = queue.enqueueNDRangeKernel(k_euler, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    //queue.finish();
    k_euler->execute();

    //err = queue.enqueueReleaseGLObjects(&enjas->cl_vbos, NULL, &event);
    cl_position->release();
    cl_color->release();
    //printf("release gl: %s\n", oclErrorString(err));

}


}
