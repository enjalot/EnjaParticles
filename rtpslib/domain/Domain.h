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


#ifndef DOMAIN_H_INCLUDED
#define DOMAIN_H_INCLUDED

#include "../structs.h"
#include <vector>
#include "../rtps_common.h"

namespace rtps {



class RTPS_EXPORT Domain
{
public:
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     */
    Domain(){};
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @param min 
     * @param max 
     */
    Domain(float4 min, float4 max);
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     */
    ~Domain();

    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @param cell_size 
     */
    void calculateCells(float cell_size);

    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getMin(){ return min; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getMax(){ return max; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getBndMin(){ return bnd_min; };
    void setBndMin(float4 new_bnd_min){ bnd_min = new_bnd_min; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getBndMax(){ return bnd_max; };
    void setBndMax(float4 new_bnd_max){ bnd_max = new_bnd_max; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getDelta() { return delta; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getRes() { return res; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
	float4 getSize() { return size; };


private:
    float4 min;
    float4 max; 
	float4 bnd_min;
	float4 bnd_max;

    float4 size;
    float4 res;
    float4 delta;

};

//-------------------------
// GORDON Datastructure for Grids. To be reconciled with Ian's
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    // particles stay within bnd
    float4          bnd_min; 
    float4          bnd_max;
    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    //float4          grid_inv_delta;
    // nb grid cells
	int 			nb_cells; 

	void print()
	{
		printf("\n----- GridParams ----\n");
		grid_min.print("grid_min"); 
		grid_max.print("grid_max"); 
		bnd_min.print("bnd_min"); 
		bnd_max.print("bnd_max"); 
		grid_res.print("grid_res"); 
		grid_size.print("grid_size"); 
		grid_delta.print("grid_delta"); 
		//grid_inv_delta.print("grid_inv_delta"); 
		printf("nb grid cells: %d\n", nb_cells);
	}
};





}
#endif
