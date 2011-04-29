#ifndef _CL_COMMON_STRUCTURES_H_
#define _CL_COMMON_STRUCTURES_H_

//----------------------------------------------------------------------
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    float4          bnd_min;
    float4          bnd_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    //float4          grid_inv_delta;

    int nb_cells;
};

#endif
