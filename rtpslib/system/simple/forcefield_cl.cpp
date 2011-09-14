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




#include "cl_structs.h"

float euclidean_distance(float4 p1, float4 p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

float magnitude(float4 v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

//normalized vector pointing from p1 to p2
float4 norm_dir(float4 p1, float4 p2)
{
    float4 dir = (float4)(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, 0.0f);
    float norm = magnitude(dir);
    if(norm > 0)
    {
        dir /= norm;
    }
    return dir;
}

float4 explode(float4 p, float4 c, float dist, float mag)
{
    float4 f = (float4)(0,0,0,0);
    float dsqr = euclidean_distance(p, c);
    if(dsqr < dist)
    {
        f = p - c;
        float norm = sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
        f /= norm;
        f *= dsqr*mag;
    }
    return f;
}
float4 implode(float4 p, float4 c, float dist, float mag)
{
    float4 f = (float4)(0,0,0,0);
    float dsqr = euclidean_distance(p, c);
    if(dsqr < dist)
    {
        f = c - p;
        float norm = sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
        f /= norm;
        f *= dsqr*mag;
    }
    return f;
}


float4 force_field(float4 p, float4 ff, float dist, float max_force)
{
    float d = euclidean_distance(p, ff);
    if(d < dist)
    {
        float4 dir = norm_dir(p, ff);
        float mag = max_force * (dist - d)/dist;
        dir *= mag;
        return dir;
    }
    return (float4)(0, 0, 0, 0);
}

float4 predator_prey(float4 p)
{
    float4 v = (float4)(0,0,0,0);
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
    return vn/6.0f;
}


void copy_local_ff(__global ForceField* ff_gl, __local ForceField* ff_loc, int one_ff, int first_ff, int last_ff)
// one_ff: nb floats in one Triangle structure
{
    int block_sz = get_local_size(0);

    // takes the values [0 to block_sz-1]
    int loc_tid = get_local_id(0);

    //ff_loc[0] = ff_gl[0];
    //ff_loc[loc_tid] = ff_gl[loc_tid];
/*
    if(loc_tid == 0)
    {
        for(int j = first_ff; j < last_ff; j++)
        {
            ff_loc[j] = ff_gl[j];
        }
    }
    */
    // first = 3, last = 7, tri = 3,4,5,6 = last - first
    int nb_ff = one_ff * (last_ff-first_ff);

// Store nb_floats (> block_sz) into shared memory
// All threads participate in the transfer
    for (int j = loc_tid; j < nb_ff; j += block_sz) {
        //if ((j+first_ff) > last_ff) break;
        ff_loc[j] = ff_gl[j+first_ff*one_ff];
    }
}

__kernel void forcefield(__global float4* pos, __global float4* vel, __global float4* force, __global float4* colors, __global ForceField* ffglob, int n_ff, __local ForceField* ffloc)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    
    //copy the forcefields to local memory
    int one_ff = 1;
    copy_local_ff(ffglob, ffloc, one_ff, 0, n_ff);
    barrier(CLK_LOCAL_MEM_FENCE);

    //external force is gravity
    //f.z += -9.8f;
    //float4 ffp = (float4)(.1f,.1f,0.1f,0.0f);
    //float dist = 1.0f;
    //float max_force = 20.0f;
    //float4 ff = force_field(p, ffloc[0].center, ffloc[0].radius, ffloc[0].max_force);
    /*
    ForceField FF = ffloc[0];
    float4 center = FF.center;
    float radius = FF.radius;
    float max_force = FF.max_force;
    float4 ff = force_field(p, center, radius, max_force);
    */
    for(int j = 0; j < n_ff; j++)
    {
        //force[i] += force_field(p, ffloc[j].center, ffloc[j].radius, ffloc[j].max_force);
        //force[i] += implode(p, ffloc[j].center, ffloc[j].radius, ffloc[j].max_force);
        force[i] += explode(p, ffloc[j].center, ffloc[j].radius, ffloc[j].max_force);
    }

    /*
        force[i] += force_field(p, ffglob[0].center, ffglob[0].radius, ffglob[0].max_force);
        force[i] += force_field(p, ffglob[1].center, ffglob[1].radius, ffglob[1].max_force);
    */
    //float4 ff = force_field(p, ffp, .3f, 10.0f);
    //force[i] += (float4)(1,0,0,0);

}
