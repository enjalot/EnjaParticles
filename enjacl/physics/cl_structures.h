// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_STRUCTURES_H_
#define _CL_STRUCTURES_H_

//----------------------------------------------------------------------
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
};
//----------------------------------------------------------------------
struct FluidParams
{
	float smoothing_length; // SPH radius
	float scale_to_simulation;
};
//----------------------------------------------------------------------


#endif
