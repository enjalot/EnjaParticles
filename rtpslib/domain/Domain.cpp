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


#include <math.h>

#include "Domain.h"

namespace rtps
{

    Domain::Domain(float4 min, float4 max)
    {
        this->bnd_min = min;
        this->bnd_max = max;
    }

    void Domain::calculateCells(float cell_size)
    {
        double s2 = 2.*cell_size;
        min = this->bnd_min - float4(s2, s2, s2, 0.);
        max = this->bnd_max + float4(s2, s2, s2, 0.);

        printf("cell size: %f\n ASDFASDFSDF\n", cell_size);

        //width of grid in each dimension
        size = float4(max.x - min.x,
                      max.y - min.y,
                      max.z - min.z,
                      0.0f);

        //number of cells in each dimension
        res = float4(ceil(size.x / cell_size),
                     ceil(size.y / cell_size),
                     ceil(size.z / cell_size),
                     0.0f);

        //width adjusted for whole number of cells
        size = float4(res.x * cell_size,
                      res.y * cell_size,
                      res.z * cell_size,
                      0.0f);

        //width of cell based on adjusted size
        delta = float4(res.x / size.x,
                       res.y / size.y,
                       res.z / size.z,
                       0.0f);
        /*
        delta = float4(size.x / res.x,
                       size.y / res.y,
                       size.z / res.z,
                       1.0f);
        */

        this->min = min;
        this->max = min + size;
        //this->max = max;


    }

    Domain::~Domain()
    {
    }

}
