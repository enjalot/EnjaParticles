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
    //pt->density = (float4)(0.,0.,0.,0.);
    //pt->color = (float4)(0.,0.,0.,0.);
    pt->color_normal = (float4)(0.,0.,0.,0.);
    pt->force = (float4)(0.,0.,0.,0.);
    //pt->surf_tens = (float4)(0.,0.,0.,0.);
    //pt->color_lapl = 0.;
    //pt->xsph = (float4)(0.,0.,0.,0.);
    //	pt->center_of_mass = (float4)(0.,0.,0.,0.);
    //	pt->num_neighbors = 0;
}


//----------------------------------------------------------------------
//Collide a fluid particle against a point (and normal). Ideally, the size of the normal
//should be proportional to the underlying surface area supported by the normal, But we not 
//have this information. 

void collision_point(PointData* pt, 
		float4 p_fluid,
		float4 v, //vel_s,  // boundary point
		float4 p_cloud,
		float4 n_cloud,  // normalized
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
        r_f = calculateRepulsionForce(n_cloud, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
        //f_f = calculateFrictionForce(v, f, nc, friction_kinetic, friction_static_limit);
    }

    pt->force += r_f;  // add friction later
    //pt->force += r_f + f_f;
}

//----------------------------------------------------------------------
// 4
inline void ForNeighborCloud(//__global float4*  vars_sorted,
						__global float4* cloud_pos, 
						__global float4* cloud_normals,
                        PointData* pt,
                        uint index_j,  // neighbor index
                        float4 p_fluid,   // position_i,
                        __constant struct GridParams* gp,
                        __constant struct SPHParams* sphp,
                        int num_cloud
                        //__constant struct CLOUDParams* cloudp
                        DEBUG_ARGS
                       )
{
    int num = sphp->num;
    //int num_cloud = cloudp->num;

    // get the particle info (in the current grid) to test against
    //float4 p_fluid = position_i; // * sphp->simulation_scale; 
    float4 p_cloud = cloud_pos[index_j] * sphp->simulation_scale; // scale NEEDED?
    float4 n_cloud = cloud_normals[index_j]; // * sphp->simulation_scale; 

    //float4 v_cloud = cloud_vel[index_j]; // * sphp->simulation_scale; 

    float4 r = (p_fluid - p_cloud);  // dist(cloud pt to fluid pt)
    r.w = 0.f; // I had stored density in 4th component
    // |r|
    float rlen = length(r);
	float4 v = (float4) (0.,0.,0.,0.);  // use cloud velocity later for friction forces

    if (rlen <= sphp->smoothing_distance)
	{
		collision_point(
			pt, // update force component
			p_fluid, 
			v, //vel_s,  
			p_cloud,
			n_cloud, 
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
						   // num ???
						   int4 cellPos,
                           //uint num,
                           float4  position_i,
						   __global float4* cloud_pos,
						   __global float4* cloud_normals,
                           __global int*       cell_indexes_start, // based on cloud points
                           __global int*       cell_indexes_end,
                           __constant struct GridParams* gp,
                           __constant struct SPHParams* sphp,
						   int num_cloud
                           //__constant struct CLOUDParams* cloudp
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
            ForNeighborCloud(cloud_pos, cloud_normals, pt, index_j, position_i, gp, sphp, num_cloud DEBUG_ARGV);
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
                                  /* uint num, */
                                  float4   position_i, 
                                  __global float4*   cloud_pos, 
                                  __global float4*   cloud_normals,   
                                  __global int*       cell_indices_start,
                                  __global int*       cell_indices_end,
                                  __constant struct GridParams* gp,
                                  //__constant struct FluidParams* fp,
                                  __constant struct SPHParams* sphp,
                                  __constant struct CLOUDParams* cloudp
                                  DEBUG_ARGS
                                  )
{
    // initialize force on particle (collisions)

	int num = sphp->num;
	int cloud_num = cloudp->num;

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
                IterateParticlesInCellCloud(pos, force, pt, /*num,*/ ipos, position_i, cloud_pos, cloud_normals, cell_indices_start, cell_indices_end, gp,/* fp,*/ sphp, cloudp DEBUG_ARGV);

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
						__global float4* cloud_pos,  
						__global float4* cloud_normals,  
						__global float4* force,

                       __global int*    cell_cloud_indexes_start,
                       __global int*    cell_cloud_indexes_end,
                       __constant struct GridParams* gp,
                       __constant struct SPHParams* sphp
                       //__constant struct CLOUDParams* cloudp  // TO ADD
                       DEBUG_ARGS
                       )
{
	return;

    // particle index
    int nb_vars = sphp->nb_vars;
    int num = sphp->num;
    //int num_cloud = cloudp->num;

    int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos[index] * sphp->simulation_scale; // SCALE needed

    //debuging
    //clf[index] = (float4)(99,0,0,0);
    //cli[index].w = 0;

    // Do calculations on particles in neighboring cells
    PointData pt;
    zeroPoint(&pt);

	// 2
	// returns force acting on particle due to neighbor cloud particles in pt.force
	// num_cloud argument is not required
    IterateParticlesInNearbyCellsCloud(pos, force, &pt, position_i, cloud_pos, cloud_normals, cell_cloud_indexes_start, cell_cloud_indexes_end, gp, sphp, num_cloud DEBUG_ARGV);
    force[index] = pt.force; 
    //clf[index].xyz = pt.force.xyz;
}

/*-------------------------------------------------------------- */
#endif

