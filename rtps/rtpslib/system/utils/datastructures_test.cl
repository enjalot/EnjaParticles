# 1 "datastructures_test.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "datastructures_test.cpp"






# 1 "cl_macros.h" 1
# 8 "datastructures_test.cpp" 2
# 1 "cl_structures.h" 1






struct GridParams
{
    float4 grid_size;
    float4 grid_min;
    float4 grid_max;


    float4 grid_res;
    float4 grid_delta;
    float4 grid_inv_delta;
    int num;
    int nb_vars;
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
};
# 9 "datastructures_test.cpp" 2


__kernel void datastructures(


     __global float4* vars_unsorted,
     __global float4* vars_sorted,
        __global uint* sort_hashes,
        __global uint* sort_indices,
        __global uint* cell_indices_start,
        __global uint* cell_indices_end,
        __constant struct GridParams* gp,
     __local uint* sharedHash
     )
{
 uint index = get_global_id(0);
 int nb_vars = gp->nb_vars;
 int num = get_global_size(0);


 if (index >= num) return;

 uint hash = sort_hashes[index];





 uint tid = get_local_id(0);


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

 uint sorted_index = sort_indices[index];
# 85 "datastructures_test.cpp"
 vars_sorted[index+1*num] = vars_unsorted[sorted_index+1*num];
 vars_sorted[index+2*num] = vars_unsorted[sorted_index+2*num];
 vars_sorted[index+0*num].x = vars_unsorted[sorted_index+0*num].x;

}
