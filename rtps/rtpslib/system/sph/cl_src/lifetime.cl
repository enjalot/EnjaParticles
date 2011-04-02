#include "cl_macros.h"
#include "cl_structs.h"

__kernel void lifetime(__global float4* pos_u, 
                        __global float4* color_u, 
                        __global float4* color_s, 
                        __global uint* sort_indices,
                        //__global float* life, 
                        //__global float4* pos_gen, 
                        //__global float4* vel_gen, 
                        float increment)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    float life = color_s[i].w;
    //decrease the life by the time step (this value could be adjusted to lengthen or shorten particle life
    life += increment;
    //if the life is 0 or less we reset the particle's values back to the original values and set life to 1
    if(life <= 0.f)
    {
        //p = pos_gen[i];
        //v = vel_gen[i];
        //life = 1.0f;    
        //this will only work once we stop using vars_unsorted (or i need to pass that in)
        //pos_u[i] = (float4)(10.0f, 10.0f, 10.0f, 1.0f);
        life = 0.f;
    }
    if(life >= 1.0)
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

    uint originalIndex = sort_indices[i];
    color_u[originalIndex] = color_s[i];




}

