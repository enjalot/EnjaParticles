#include "../Kinect.h"
#include <math.h>
#include "../ForceField.h"

namespace rtps {

void Kinect::loadForceField()
{

    std::vector<ForceField> tff(max_forcefields);
    //cl_forcefield = Buffer<ForceField>(ps->cli, tff, CL_MEM_WRITE_ONLY);
    cl_forcefield = Buffer<ForceField>(ps->cli, tff);
    //cl_forcefield = Buffer<ForceField>(ps->cli, forcefields);

    std::string path(SIMPLE_CL_SOURCE_DIR);
    path += "/forcefield_cl.cl";
    k_forcefield = Kernel(ps->cli, path, "forcefield");
  
    //k_forcefield.setArg(0, cl_position.getDevicePtr());
    k_forcefield.setArg(0, cl_position.getDevicePtr());
    k_forcefield.setArg(1, cl_velocity.getDevicePtr());
    k_forcefield.setArg(2, cl_force.getDevicePtr());
    k_forcefield.setArg(3, cl_color.getDevicePtr());   //forcefields
    k_forcefield.setArg(4, cl_forcefield.getDevicePtr());   //forcefields
    

   }

void Kinect::loadForceFields(std::vector<ForceField> ff)
{
    glFinish();
    printf("LOAD forcefields: %d\n", forcefields_enabled);
    if (!forcefields_enabled)
        return;
    int n_forcefields = ff.size();
    printf("n forcefields: %d\n", n_forcefields);
    //load forcefields into cl buffer
    //ForceField is a struct that ends up being 4 float4s
    //cl_forcefields = cl::Buffer(context, CL_MEM_WRITE_ONLY, ff_size, NULL, &err);
    //err = queue.enqueueWriteBuffer(cl_forcefields, CL_TRUE, 0, ff_size, &ff[0], NULL, &event);
    for(int i = 0; i < ff.size(); i++)
    {
        printf("FF: %f %f\n", ff[i].center.x, ff[i].center.y);
    }
    cl_forcefield.copyToDevice(ff, 0);
    
    k_forcefield.setArg(5, n_forcefields);   //number of forcefields

    printf("sizeof(ForceField) = %d\n", (int) sizeof(ForceField));


    size_t max_loc_memory = 1024 << 4;  // 16k bytes local memory on mac
    int max_ff = max_loc_memory / sizeof(ForceField);
    //max_ff = n_forcefields;
    //max_ff = 128; // fits in cache
    max_ff = 10; // fits in cache
    printf("max_ff= %d\n", max_ff);
        
    size_t sz = max_ff*sizeof(ForceField);
    printf("sz= %zd bytes\n", sz);

    k_forcefield.setArgShared(6, sz);   //number of forcefields


}


void Kinect::cpuForceField()
{
    //float4 c = forcefields[0].center;
    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];


        for(int j = 0; j < forcefields.size(); j++)
        {
            //float4 pp = predator_prey(p);
            //float4 pp = runge_kutta(p, h);
            //v = pp;
            f += force_field(p, forcefields[j].center, forcefields[j].radius, forcefields[j].max_force);
        }
        //velocities[i] = v;
        forces[i] = f;
    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}
