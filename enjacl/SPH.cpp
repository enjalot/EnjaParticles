#include "SPH.h"
#include "util.h"

#include "physics/sph/density.cl"
#include "physics/sph/pressure.cl"
#include "physics/sph/viscosity.cl"
#include "physics/sph/collision_wall.cl"
#include "physics/sph/euler.cl"
const std::string SPH::sources[] = {
    density_program_source,
    pressure_program_source,
    viscosity_program_source,
    collision_wall_program_source,
    euler_program_source
};


SPH::SPH(EnjaParticles *enj)
{
    enjas = enj;

    //we take some variables that EnjaParticles have already prepared
    context = enjas->context;
    queue = enjas->queue;
    num = enjas->num;

    //create buffers
    cl_density = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float)*num, NULL, &err); //float array
    cl_force = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Vec4)*num, NULL, &err); //float4 array

    //initialize force
    // don't need to because it gets set to 0 first in pressure calculation
    //std::vector<float> dens(num);
    //std::vector<Vec4> forc(num);
    //std::fill(dens.begin(), dens.end(), 0.0f);
    //Vec4 f(0.0f, 0.0f, -9.8f, 0.0f);
    //std::fill(forc.begin(), forc.end(), f);
    //err = queue.enqueueWriteBuffer(cl_force, CL_TRUE, 0, sizeof(Vec4)*num, &forc[0], NULL, &event);

    //cl_velocity = cl::Buffer(context, CL_MEM_READ_WRITE, 4*sizeof(float)*num, NULL, &err); //float4 array
    cl_velocity = enjas->cl_velocities;

    //create kernels
    printf("create density kernel\n");
    k_density = enjas->loadKernel(sources[DENSITY], "density");
    k_density.setArg(0, enjas->cl_vbos[0]);
    k_density.setArg(1, cl_density);

    printf("create pressure kernel\n");
    k_pressure = enjas->loadKernel(sources[PRESSURE], "pressure");
    k_pressure.setArg(0, enjas->cl_vbos[0]);
    k_pressure.setArg(1, cl_density);
    k_pressure.setArg(2, cl_force);

    printf("create viscosity kernel\n");
    k_viscosity = enjas->loadKernel(sources[VISCOSITY], "viscosity");
    k_viscosity.setArg(0, enjas->cl_vbos[0]);
    k_viscosity.setArg(1, cl_velocity);

    printf("create collision_wall kernel\n");
    k_collision_wall = enjas->loadKernel(sources[COLLISION_WALL], "collision_wall");
    k_collision_wall.setArg(0, enjas->cl_vbos[0]);
    k_collision_wall.setArg(1, cl_velocity);
    k_collision_wall.setArg(2, cl_force);
    //k_collision_wall.setArg(2, cl_walls); //need to make a data structure for grid walls

    printf("create euler kernel\n");
    //could generalize this to other integration methods later (leap frog, RK4)
    k_euler = enjas->loadKernel(sources[EULER], "euler");
    k_euler.setArg(0, enjas->cl_vbos[0]);
    k_euler.setArg(1, cl_velocity);
    k_euler.setArg(2, cl_force);
    k_euler.setArg(3, .01f); //time step (should be set from settings)



}

SPH::~SPH()
{
}

void SPH::update()
{
    //call kernels
    //add timings
    glFinish();
    err = queue.enqueueAcquireGLObjects(&enjas->cl_vbos, NULL, &event);
    //printf("acquire: %s\n", oclErrorString(err));
    queue.finish();
    
    //pressure
    err = queue.enqueueNDRangeKernel(k_pressure, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    //collision
    err = queue.enqueueNDRangeKernel(k_collision_wall, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    //euler integration
    err = queue.enqueueNDRangeKernel(k_euler, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    queue.finish();

    err = queue.enqueueReleaseGLObjects(&enjas->cl_vbos, NULL, &event);
    //printf("release gl: %s\n", oclErrorString(err));
    queue.finish();

}
