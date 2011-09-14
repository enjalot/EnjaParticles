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


#ifndef _CLHASH_CL_H_
#define _CLHASH_CL_H_



//----------------------------------------------------------------------
// find the grid cell from a position in world space
// WHY static?
int4 calcGridCell(float4 p, float4 grid_min, float4 grid_delta)
{
    // subtract grid_min (cell position) and multiply by delta
    //return make_int4((p-grid_min) * grid_delta);

    //float4 pp = (p-grid_min)*grid_delta;
    float4 pp;
    pp.x = (p.x-grid_min.x)*grid_delta.x;
    pp.y = (p.y-grid_min.y)*grid_delta.y;
    pp.z = (p.z-grid_min.z)*grid_delta.z;
    pp.w = (p.w-grid_min.w)*grid_delta.w;




    int4 ii;
    ii.x = (int) pp.x;
    ii.y = (int) pp.y;
    ii.z = (int) pp.z;
    ii.w = (int) pp.w;
    return ii;
}

//----------------------------------------------------------------------
int calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges
                  //,__global float4* fdebug,
                  //__global int4* idebug
                 )
{
    // each variable on single line or else STRINGIFY DOES NOT WORK
    int gx;
    int gy;
    int gz;

    if (wrapEdges)
    {
        int gsx = (int)floor(grid_res.x);
        int gsy = (int)floor(grid_res.y);
        int gsz = (int)floor(grid_res.z);

        //          //power of 2 wrapping..
        //          gx = gridPos.x & gsx-1;
        //          gy = gridPos.y & gsy-1;
        //          gz = gridPos.z & gsz-1;

        // wrap grid... but since we can not assume size is power of 2 we can't use binary AND/& :/
        gx = gridPos.x % gsx;
        gy = gridPos.y % gsy;
        gz = gridPos.z % gsz;
        if (gx < 0) gx+=gsx;
        if (gy < 0) gy+=gsy;
        if (gz < 0) gz+=gsz;
    }
    else
    {
        gx = gridPos.x;
        gy = gridPos.y;
        gz = gridPos.z;
    }

    //We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
    //this space filling curve is then simply:
    // index = x + y*width + z*width*height
    //This means that we process the grid structure in "depth slice" order, and
    //each such slice is processed in row-column order.

    //int index = get_global_id(0);
    //fdebug[index] = grid_res;
    //idebug[index] = (int4)(gx,gy,gz,1.);
    //idebug[index] = (gz*grid_res.y + gy) * grid_res.x + gx; 

    // uint(-3) = 0   (so hash is never less than zero)
    // But if particle leaves boundary to the right of the grid, the hash
    // table can go out of bounds and the code might crash. This can happen
    // either if the boundary does not catch the particles or if the courant
    // condition is violated and the code goes unstable. 
    //  ^ this is resolved by checking hash

    //int ret = (gz*grid_res.y + gy) * grid_res.x + gx; 
    //return ret;
    return (gz*grid_res.y + gy) * grid_res.x + gx; 
}
#endif
