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


#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_




/*----------------------------------------------------------------------*/

#include "cl_hash.h"
#include "cl_macros.h"
#include "cl_structs.h"
//Contains all of the Smoothing Kernels for SPH
//#include "cl_kernels.h"
#include "cl_collision.h"

// for zeroPoint (redefine zeroPoint instead)
//#include "cl_neighbors.h" 

//#define ARGS int pts_in_cloud,  __global float4* pos,  __global float* cloud_pos,  __global float* cloud_normals,  __global float4* force

//----------------------------------------------------------------------
void zeroPoint(PointData* pt)
{
    pt->color_normal = (float4)(0.,0.,0.,0.);
    pt->force = (float4)(0.,0.,0.,0.);
}
//----------------------------------------------------------------------
//Collide a fluid particle against a point (and normal). Ideally, the size of the normal
//should be proportional to the underlying surface area supported by the normal, But we not 
//have this information. 

void collision_point(PointData* pt, 
		float4 p_fluid,
		float4 v_fluid, //vel_s,  // boundary point
		float4 p_cloud,
		float4 n_cloud,  // normalized
		float4 v_cloud,  
		__constant struct GridParams* gp,
		__constant struct SPHParams* sphp)
{
    float4 r_f = (float4) (0.,0.,0.,0.);
    float4 f_f = (float4) (0.,0.,0.,0.);

    float friction_kinetic = 0.0f;
    float friction_static_limit = 0.0f;

	// dd: distance between p and pc
    float4 dist = p_fluid - p_cloud;  // computed before
	dist.w = 0;
	float dist1 = sqrt(dot(dist, dist));
    float diff = sphp->boundary_distance - dist1;

	 // the boundary surface normal nc is pointing into the fluid 

    if (diff > sphp->EPSILON)
    {
		// assume n_cloud has unit length
		float4 v_cloud = .003; // TEMPORARY
        r_f = calculateRepulsionForce(n_cloud, v_fluid-v_cloud, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
        //f_f = calculateFrictionForce(v, f, nc, friction_kinetic, friction_static_limit);
		;
    }

    pt->force += r_f;  // add friction later
    //pt->force += r_f + f_f;
}

//----------------------------------------------------------------------
// 4
inline void ForNeighborCloud(//__global float4*  vars_sorted,
						__global float4* cloud_pos, 
						__global float4* cloud_normals,
						__global float4* cloud_velocity,
                        PointData* pt,
                        uint index_j,  // neighbor index
                        float4 p_fluid,   // position_i,
                        float4 v_fluid,   // velocity_i
                        __constant struct GridParams* gp,
                        __constant struct SPHParams* sphp,
                        int num_cloud
                        DEBUG_ARGS
                       )
{
    int num = sphp->num;

    // get the particle info (in the current grid) to test against
    //float4 p_fluid = position_i; // * sphp->simulation_scale; 
	// cloud_pos always in world coord
    float4 p_cloud = cloud_pos[index_j] * sphp->simulation_scale; 
    float4 n_cloud = cloud_normals[index_j]; // * sphp->simulation_scale; 
    float4 v_cloud = cloud_velocity[index_j]; // * sphp->simulation_scale; 

    //float4 v_cloud = cloud_vel[index_j]; // * sphp->simulation_scale; 

    float4 r = (p_fluid - p_cloud);  // dist(cloud pt to fluid pt)
    r.w = 0.f; // I had stored density in 4th component
    // |r|
    float rlen = length(r);

    if (rlen <= sphp->smoothing_distance)
	{
		collision_point(
			pt, // update force component
			p_fluid, 
			v_fluid,
			p_cloud,
			n_cloud, 
			v_cloud, 
			gp,
			sphp);
	}
}

//----------------------------------------------------------------------
// 3
void IterateParticlesInCellCloud(
						   __global float4* pos,
						   __global float4* force,
						   PointData* pt,
						   int4 cellPos,
                           float4  position_i, // of fluid
                           float4  velocity_i, // of fluid
						   __global float4* cloud_pos,
						   __global float4* cloud_normals,
						   __global float4* cloud_velocity,
                           __global int* cell_indexes_start, // based on cloud points
                           __global int*  cell_indexes_end,
                           __constant struct GridParams* gp,
                           __constant struct SPHParams* sphp,
						   int num_cloud
                           DEBUG_ARGS
                           )
{
    // get hash (of position) of current cell
    uint cellHash = calcGridHash(cellPos, gp->grid_res, false);
    
    //need to check cellHash to make sure its not out of bounds
    if(cellHash >= gp->nb_cells)
    {
        return;
    }
    //even with cellHash in bounds we are still getting out of bounds indices

    /* get start/end positions for this cell/bucket */
    uint startIndex = FETCH(cell_indexes_start,cellHash);
    /* check cell is not empty
     * WHERE IS 0xffffffff SET?  NO IDEA ************************
     */
    if (startIndex != 0xffffffff)
    {
        uint endIndex = FETCH(cell_indexes_end, cellHash);

        /* iterate over cloud particles in this cell */
        for (uint index_j=startIndex; index_j < endIndex; index_j++)
        {
            //***** UPDATE pt (sum) (4)
            ForNeighborCloud(cloud_pos, cloud_normals, cloud_velocity, pt, index_j, position_i, velocity_i, gp, sphp, num_cloud DEBUG_ARGV);
        }
    }
}

//----------------------------------------------------------------------

// 2
void IterateParticlesInNearbyCellsCloud(
                                  //__global float4* vars_sorted,
								  __global float4*  pos, 
								  __global float4* force,
                                  PointData* pt,
                                  float4   position_i, 
                                  float4   velocity_i, 
                                  __global float4*   cloud_pos, 
                                  __global float4*   cloud_normals,   
                                  __global float4*   cloud_velocity,   
                                  __global int*       cell_indices_start,
                                  __global int*       cell_indices_end,
                                  __constant struct GridParams* gp,
                                  __constant struct SPHParams* sphp,
                                  int cloud_num
                                  DEBUG_ARGS
                                  )
{
    // initialize force on particle (collisions)

	int num = sphp->num;

    // get cell in grid for the given position
    int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);

    // iterate through the 3^3 cells in and around the given position
    // can't unroll these loops, they are not innermost 
    //TODO bug in that we don't take into account cells on edge of grid!
    for (int z=cell.z-1; z<=cell.z+1; ++z)
    {
        for (int y=cell.y-1; y<=cell.y+1; ++y)
        {
            for (int x=cell.x-1; x<=cell.x+1; ++x)
            {
                int4 ipos = (int4) (x,y,z,1);

                // **** SUMMATION/UPDATE
				// 3
                IterateParticlesInCellCloud(pos, force, pt, ipos, position_i, velocity_i, cloud_pos, cloud_normals, cloud_velocity, cell_indices_start, cell_indices_end, gp, sphp, cloud_num DEBUG_ARGV);

                //barrier(CLK_LOCAL_MEM_FENCE); // DEBUG
                // SERIOUS PROBLEM: Results different than results with cli = 5 (bottom of this file)
            }
        }
    }
}
//----------------------------------------------------------------------

//--------------------------------------------------------------
// compute forces on particles interaction with a boundary formed from 
// other particles. 

// 1
__kernel void collision_cloud(
						int num_cloud,
						__global float4* pos,  
						__global float4* vel,  
						__global float4* cloud_pos,  
						__global float4* cloud_normals,  
						__global float4* cloud_velocity,  
						__global float4* force,

                        __global int*    cell_cloud_indexes_start,
                        __global int*    cell_cloud_indexes_end,
                        __constant struct GridParams* gp,
                        __constant struct SPHParams* sphp
                        DEBUG_ARGS
                       )
{
    // particle index
    //int nb_vars = sphp->nb_vars;
    int num = sphp->num;

    int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos[index] * sphp->simulation_scale;
    float4 velocity_i = vel[index]; // in simulation scale

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

	// 2
	// returns force acting on particle due to neighbor cloud particles 
	// in pt.force
	// num_cloud argument is not required
    IterateParticlesInNearbyCellsCloud(pos, force, &pt, position_i, velocity_i, cloud_pos, cloud_normals, cloud_velocity, cell_cloud_indexes_start, cell_cloud_indexes_end, gp, sphp, num_cloud DEBUG_ARGV);

	// must somehow scale according to nb points in neighborhood
	// pt.force: sum of all the boundary forces acting on the particle

	float fact = 1.;        // scale the boundary force (arbitrary factor. Not satisfactory. GE)

	pt.force.x *= fact;
	pt.force.y *= fact;
	pt.force.z *= fact;

	float ff = length(pt.force);

	if (ff > 1.e-4) {
    	force[index] = pt.force;
    	//force[index] += pt.force;
	}

	// original
    //force[index] += pt.force;

	// what happens if force on boundary particle is cancelled? 
    //clf[index].xyz = pt.force.xyz;
}

/*-------------------------------------------------------------- */
#endif

