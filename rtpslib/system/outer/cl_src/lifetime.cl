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


#include "cl_macros.h"
#include "cl_structs.h"

__kernel void lifetime( int num,
                        float increment,
                        __global float4* pos_u, 
                        __global float4* color_u, 
                        __global float4* color_s, 
                        __global uint* sort_indices
                        DEBUG_ARGS
                        )
{
    //get our index in the array
    unsigned int i = get_global_id(0);
    if(i >= num) return;

    float life = color_s[i].w;
    //decrease the life by the time step (this value could be adjusted to lengthen or shorten particle life
    life += increment;
    //if the life is 0 or less we reset the particle's values back to the original values and set life to 1
    if(life <= 0.0f)
    {
        //p = pos_gen[i];
        //v = vel_gen[i];
        //life = 1.0f;    
        //this will only work once we stop using vars_unsorted (or i need to pass that in)
        pos_u[i] = (float4)(100.0f, 100.0f, 100.0f, 1.0f);
        //pos_u[i] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        life = 0.0f;
    }
    if(life >= 1.0f)
    {
        life = 1.0f;
    }
    float alpha = life;
    
    //you can manipulate the color based on properties of the system
    //here we adjust the alpha
    /*
    color_s[i].x = alpha;
    color_s[i].y = alpha;
    color_s[i].z = alpha;
    */
    color_s[i].w = alpha;

    //uint originalIndex = sort_indices[i];
    //color_u[originalIndex] = color_s[i];
    color_u[i] = color_s[i];
    //clf[i] = color_u[originalIndex];
    //cli[i] = sort_indices[i];
    //clf[i] = color_s[i];
    //color_u[originalIndex] = (float4)(0.f, 0.f, 1.f, 1.f);
    //color_u[i] = (float4)(0.f, 0.f, 1.f, 1.f);


}

