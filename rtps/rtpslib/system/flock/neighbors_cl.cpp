#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


/* TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 
*/

/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"
#include "cl_neighbors.h"
#include "cl_hash.h"


	/*--------------------------------------------------------------*/
	/* Iterate over particles found in the nearby cells (including cell of position_i)
	*/
	void IterateParticlesInCell(
		__global float4*    vars_sorted,
		PointData* pt,
        uint num,
		int4 	cellPos,
		uint 	index_i,
		float4 	position_i,
		__global int* 		cell_indexes_start,
		__global int* 		cell_indexes_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct FLOCKParams* flockp
		DEBUG_ARGS
    )
	{
		// get hash (of position) of current cell
		uint cellHash = calcGridHash(cellPos, gp->grid_res, false);

		/* get start/end positions for this cell/bucket */
		uint startIndex = FETCH(cell_indexes_start,cellHash);
		/* check cell is not empty
		 * WHERE IS 0xffffffff SET?  NO IDEA ************************
		 */
		if (startIndex != 0xffffffff) {	   
			uint endIndex = FETCH(cell_indexes_end, cellHash);

			/* iterate over particles in this cell */
			for(uint index_j=startIndex; index_j < endIndex; index_j++) {			
#if 1
				//***** UPDATE pt (sum)
				ForPossibleNeighbor(vars_sorted, pt, num, index_i, index_j, position_i, gp, /*fp,*/ flockp DEBUG_ARGV);
#endif
			}
		}
	}

	/*--------------------------------------------------------------*/
	/* Iterate over particles found in the nearby cells (including cell of position_i) 
	 */
	void IterateParticlesInNearbyCells(
		__global float4* vars_sorted,
		PointData* pt,
        uint num,
		int 	index_i, 
		float4   position_i, 
		__global int* 		cell_indices_start,
		__global int* 		cell_indices_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct FLOCKParams* flockp
		DEBUG_ARGS
		)
	{
		// initialize force on particle (collisions)

		// get cell in grid for the given position
		//int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_inv_delta);
		int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost 
		for(int z=cell.z-1; z<=cell.z+1; ++z) {
			for(int y=cell.y-1; y<=cell.y+1; ++y) {
				for(int x=cell.x-1; x<=cell.x+1; ++x) {
					int4 ipos = (int4) (x,y,z,1);

					// **** SUMMATION/UPDATE
					IterateParticlesInCell(vars_sorted, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ flockp DEBUG_ARGV);

				//barrier(CLK_LOCAL_MEM_FENCE); // DEBUG
				// SERIOUS PROBLEM: Results different than results with cli = 5 (bottom of this file)
				}
			}
		}
	}

	//----------------------------------------------------------------------
//--------------------------------------------------------------
// compute forces on particles

__kernel void neighbors(
				__global float4* vars_sorted,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
				__constant struct GridParams* gp,
				//__constant struct FluidParams* fp, 
				__constant struct FLOCKParams* flockp 
				DEBUG_ARGS
				)
{
    // particle index
	int nb_vars = flockp->nb_vars;
	int num = flockp->num;
    //int numParticles = get_global_size(0);
    //int num = get_global_size(0);


	int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos(index);

    //debuging
    cli[index].w = 0;


    // Do calculations on particles in neighboring cells
	PointData pt;
	zeroPoint(&pt);

	if (flockp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		density(index) = flockp->wpoly6_coef * pt.density.x;
        clf[index].w = density(index);
		// code reaches this point on first call
	}
	if (flockp->choice == 1) { // update force
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		force(index) = pt.force; // Does not seem to maintain value into euler.cl
        clf[index].xyz = pt.force.xyz;
		xflock(index) = flockp->wpoly6_coef * pt.xflock;
	}

	if (flockp->choice == 2) { // update surface tension (NOT DEBUGGED)
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ flockp DEBUG_ARGV);
		float norml = length(pt.color_normal);
		if (norml > 1.) {
			float4 stension = -0.3f * pt.color_lapl * pt.color_normal / norml;
			force(index) += stension; // 2 memory accesses (NOT GOOD)
		}
	}
	if (flockp->choice == 3) { // denominator in density normalization
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ flockp DEBUG_ARGV);

		density(index) /= pt.density.y;
	}
	
	/*if (flockp->choice == 4) { //Extract surface particles
		IterateParticlesInNearbyCells(vars_sorted,&pt,num,index, position_i, cell_indexes_start, cell_indexes_end, gp, flockp DEBUG_ARGV);
		
		pt.center_of_mass = pt.center_of_mass/(float) pt.num_neighbors;
		float4 dist = pos(index)-pt.center_of_mass;
		dist.w = 0;
		if(pt.num_neighbors < 5 ||
			sqrt(dot(dist,dist)) > flockp->surface_threshold)	
			surface(index) = (float4){1.0,1.0,1.0,1.0};
		else
			surface(index) = (float4){0.0,0.0,0.0,0.0};
	}*/
}

/*-------------------------------------------------------------- */
#endif

