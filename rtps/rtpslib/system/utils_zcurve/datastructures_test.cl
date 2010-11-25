# 1 "datastructures_test.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "datastructures_test.cpp"






# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 8 "datastructures_test.cpp" 2
# 1 "cl_structures.h" 1






struct GPUReturnValues
{
 int compact_size;
};

struct CellOffsets
{
 int4 offsets[32];
};



typedef struct PointData
{


 float4 density;
 float4 color;
 float4 color_normal;
 float4 color_lapl;
 float4 force;
 float4 surf_tens;
 float4 xsph;
} PointData;

struct GridParamsScaled

{
    float4 grid_size;
    float4 grid_min;
    float4 grid_max;
    float4 bnd_min;
    float4 bnd_max;


    float4 grid_res;
    float4 grid_delta;
    float4 grid_inv_delta;
    int num;
    int nb_vars;
    int nb_points;
 int4 expo;
 int4 shift[27];
};

struct GridParams
{
    float4 grid_size;
    float4 grid_min;
    float4 grid_max;
    float4 bnd_min;
    float4 bnd_max;


    float4 grid_res;
    float4 grid_delta;
    float4 grid_inv_delta;
    int num;
    int nb_vars;
    int nb_points;
 int4 expo;
 int4 shift[27];
};

struct FluidParams
{
 float smoothing_length;
 float scale_to_simulation;


 float friction_coef;
 float restitution_coef;
 float damping;
 float shear;
 float attraction;
 float spring;
 float gravity;
 int choice;
};


struct SPHParams
{
    float4 grid_min;
    float4 grid_max;
    float grid_min_padding;
    float grid_max_padding;
    float mass;
    float rest_distance;
    float rest_density;
    float smoothing_distance;
    float particle_radius;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;
    float K;
 float dt;


 float wpoly6_coef;
 float wpoly6_d_coef;
 float wpoly6_dd_coef;
 float wspike_coef;
 float wspike_d_coef;
 float wspike_dd_coef;
 float wvisc_coef;
 float wvisc_d_coef;
 float wvisc_dd_coef;

};
# 9 "datastructures_test.cpp" 2


__kernel void datastructures(
     __global float4* vars_unsorted,
     __global float4* vars_sorted,
        __global int* sort_hashes,
        __global int* sort_indices,
        __global int* cell_indices_start,
        __global int* cell_indices_end,
        __global int* cell_indices_nb,
        __constant struct SPHParams* sphp,
        __constant struct GridParams* gp,
     __local int* sharedHash
     )
{
 int index = get_global_id(0);
 int num = get_global_size(0);



 if (index >= num) return;

 int hash = sort_hashes[index];





 int tid = get_local_id(0);


 sharedHash[tid+1] = hash;

 if (index > 0 && tid == 0) {

  sharedHash[0] = sort_hashes[index-1];
 }


 barrier(CLK_LOCAL_MEM_FENCE);
# 57 "datastructures_test.cpp"
 if ((index == 0 || hash != sharedHash[tid]) )
 {
  cell_indices_start[hash] = index;
  if (index > 0) {
   cell_indices_end[sharedHash[tid]] = index;
  }
 }


 if (index == num - 1) {
  cell_indices_end[hash] = index + 1;
 }

 int sorted_index = sort_indices[index];
# 84 "datastructures_test.cpp"
 vars_sorted[index+1*num] = vars_unsorted[sorted_index+1 *num] * sphp->simulation_scale;
 vars_sorted[index+2*num] = vars_unsorted[sorted_index+2 *num];
 vars_sorted[index+8*num] = vars_unsorted[sorted_index+8*num];


}
