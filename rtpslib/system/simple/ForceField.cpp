/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include "Simple.h"
#include <math.h>

namespace rtps {

void Simple::loadForceField()
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

void Simple::loadForceFields(std::vector<ForceField> ff)
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


float distance(float4 p1, float4 p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

float length(float4 v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

//normalized vector pointing from p1 to p2
float4 norm_dir(float4 p1, float4 p2)
{
    float4 dir = float4(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, 0.0f);
    float norm = length(dir);
    if(norm > 0)
    {
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    }
    return dir;
}

float4 predator_prey(float4 p)
{
    float4 v = float4(0,0,0,0);
    int a1 = 2;
    int a2 = 2;
    int b1 = 1;
    int b2 = 1;
    v.x = a1*p.x - b1*p.x*p.y;
    v.y = -a2*p.y + b2*p.y*p.x;
    //v.x = a1 - b1*p.y;
    //v.y = -a2 + b2*p.x;
    return v;
}

float4 runge_kutta(float4 yn, float h)
{
    float4 k1 = predator_prey(yn); 
    float4 k2 = predator_prey(yn + .5f*h*k1);
    float4 k3 = predator_prey(yn + .5f*h*k2);
    float4 k4 = predator_prey(yn + h*k3);

    float4 vn = (k1 + 2.f*k2 + 2.f*k3 + k4);
    return (1./6.)*vn;
    
    /*
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
    */
}



float4 force_field(float4 p, float4 ff, float dist, float max_force)
{
    float d = distance(p, ff);
    //printf("distance: %f, dist %f\n", d, dist);
    if(d < dist)
    {
        float4 dir = norm_dir(p, ff);
        float mag = max_force * (dist - d)/dist;
        dir.x *= mag;
        dir.y *= mag;
        dir.z *= mag;
        printf("forcefield: %f, %f, %f\n", dir.x, dir.y, dir.z);
        return dir;
    }
    return float4(0, 0, 0, 0);
}

void Simple::cpuForceField()
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
