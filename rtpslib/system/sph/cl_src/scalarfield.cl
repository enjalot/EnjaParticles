#include "cl_macros.h"
#include "cl_structs.h"


__kernel void scalarfield(
                      __global float4* pos_u,
                      __global float* density,
                      __write_only image3D_t s_field,
                      //__read_only int4 mc_cells,
                      __constant struct GridParams grid_param,
                      __constant struct SPHParams* sphp)
{
    sampler_t smplr = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    int4 imgDims = get_image_dim(in);

    unsigned int i = get_global_id(0);
    int num = sphp->num;
    if (i >= num) return;

    float4 normalized_pos = float4((pos_u[i]-grid_param->grid_min).xyz/(grid_param->grid_max-grid_param_min).xyz,0.0);
    //int4 ind = int4(floor(pos_u[i].x*mc_cells.x),floor(pos_u[i].y*mc_cells.y),floor(pos_u[i].z*mc_cells.z),0);
    int4 ind = floor(normailized_pos*imgDims);
    //float4 cur_dens = read_imagef(s_field,smplr,ind);
    write_imagef(s_field,ind,/*cur_dens+*/float4(density[i],0.0,0.0,1.0));
}
