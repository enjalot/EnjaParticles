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


#ifndef UNIFORMGRID_H_INCLUDED
#define UNIFORMGRID_H_INCLUDED

#include "../structs.h"
#include "../rtps_common.h"

namespace rtps
{

    class RTPS_EXPORT UniformGrid
    {
    public:
        UniformGrid()
        {
        };
        UniformGrid(float4 min, float4 max, float cell_size);
        ~UniformGrid();

        void make_cube(float4 *positions, float spacing, int num);
        void make_column(float4 *positions, float spacing, int num);
        void make_dam(float4 *positions, float spacing, int num);
        int make_line(float4 *positions, float spacing, int num);

        float4 getMin()
        {
            return min;
        };
        float4 getMax()
        {
            return max;
        };

    private:
        float4 min;
        float4 max; 

        float4 size;
        float4 res;
        float4 delta;


    };

}
#endif
