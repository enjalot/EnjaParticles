#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_


/* TO BE INCLUDED FROM OTHER FILES. In OpenCL, I believe that all device code
// must be in the same file as the kernel using it. 
*/

/*----------------------------------------------------------------------*/

#include "cl_macros.h"
#include "cl_structs.h"
#include "cl_neighbors.h"


/*--------------------------------------------------------------*/
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

/*--------------------------------------------------------------*/
uint calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges)
{
	// each variable on single line or else STRINGIFY DOES NOT WORK
	int gx;
	int gy;
	int gz;

	if(wrapEdges) {
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
		if(gx < 0) gx+=gsx;
		if(gy < 0) gy+=gsy;
		if(gz < 0) gz+=gsz;
	} else {
		gx = gridPos.x;
		gy = gridPos.y;
		gz = gridPos.z;
	}


	//We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
	//this space filling curve is then simply:
	// index = x + y*width + z*width*height
	//This means that we process the grid structure in "depth slice" order, and
	//each such slice is processed in row-column order.

	return (gz*grid_res.y + gy) * grid_res.x + gx; 
}

	/*--------------------------------------------------------------*/
	/* Iterate over particles found in the nearby cells (including cell of position_i)
	*/
	void IterateParticlesInCell(
		__global float4*    vars_sorted,
		PointData* pt,
		uint 	numParticles,
		int4 	cellPos,
		uint 	index_i,
		float4 	position_i,
		__global int* 		cell_indexes_start,
		__global int* 		cell_indexes_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct SPHParams* sphp
		//DUMMY_ARGS
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
				ForPossibleNeighbor(vars_sorted, pt, numParticles, index_i, index_j, position_i, gp, /*fp,*/ sphp /*ARGS*/);
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
		int		numParticles, // on Linux, remove __constant
		int 	index_i, 
		float4   position_i, 
		__global int* 		cell_indices_start,
		__global int* 		cell_indices_end,
		__constant struct GridParams* gp,
		//__constant struct FluidParams* fp,
		__constant struct SPHParams* sphp
		//DUMMY_ARGS
		)
	{
		// initialize force on particle (collisions)

		// get cell in grid for the given position
		int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_inv_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost 
		for(int z=cell.z-1; z<=cell.z+1; ++z) {
			for(int y=cell.y-1; y<=cell.y+1; ++y) {
				for(int x=cell.x-1; x<=cell.x+1; ++x) {
					int4 ipos = (int4) (x,y,z,1);

					// **** SUMMATION/UPDATE
					IterateParticlesInCell(vars_sorted, pt, numParticles, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp,/* fp,*/ sphp /*ARGS*/);

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
				__constant struct SPHParams* sphp 
				//DUMMY_ARGS
				)
{
    // particle index
	int nb_vars = sphp->nb_vars;
	int numParticles = sphp->num;


	int index = get_global_id(0);
    if (index >= numParticles) return;

    float4 position_i = pos(index);

    // Do calculations on particles in neighboring cells


	PointData pt;
	zeroPoint(&pt);

	if (sphp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(vars_sorted, &pt, numParticles, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp /*ARGS*/);
		density(index) = sphp->wpoly6_coef * pt.density.x;
		// code reaches this point on first call
	}
	if (sphp->choice == 1) { // update force
    	IterateParticlesInNearbyCells(vars_sorted, &pt, numParticles, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ sphp /*ARGS*/);
		force(index) = pt.force; // Does not seem to maintain value into euler.cl
		xsph(index) = sphp->wpoly6_coef * pt.xsph;
		// SERIOUS PROBLEM: Results different than results with cli = 4 (bottom of this file)
	}
	if (sphp->choice == 2) { // update surface tension (NOT DEBUGGED)
    	IterateParticlesInNearbyCells(vars_sorted, &pt, numParticles, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ sphp /*ARGS*/);
		float norml = length(pt.color_normal);
		if (norml > 1.) {
			float4 stension = -0.3f * pt.color_lapl * pt.color_normal / norml;
			force(index) += stension; // 2 memory accesses (NOT GOOD)
		}
	}
	if (sphp->choice == 3) { // denominator in density normalization
    	IterateParticlesInNearbyCells(vars_sorted, &pt, numParticles, index, position_i, cell_indexes_start, cell_indexes_end, gp, /*fp,*/ sphp /*ARGS*/);

		// NOT WORKING. NEED DEBUG STATEMENTS
		density(index) /= pt.density.y;
	}
}

/*-------------------------------------------------------------- */
#endif

