# 1 "euler_cl.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "euler_cl.cpp"

# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 3 "euler_cl.cpp" 2
# 1 "cl_structures.h" 1







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
# 4 "euler_cl.cpp" 2



__kernel void ge_euler(
  __global int* sort_indices,
  __global float4* vars_unsorted,
  __global float4* vars_sorted,

  __global float4* positions,
  __constant struct SPHParams* params,
  float dt)
{
    unsigned int i = get_global_id(0);
 int num = get_global_size(0);

    float4 p = vars_sorted[i+1*num];
    float4 v = vars_sorted[i+2*num];
    float4 f = vars_sorted[i+3*num];





    f.z += -9.8f;



    float speed = length(f);
    if(speed > 600.0f)

    {
        f *= 600.0f/speed;
    }


 float dtt = dt;

    v += dtt*f;
    p += dtt*v / params->simulation_scale;
    p.w = 1.0f;
# 52 "euler_cl.cpp"
        uint originalIndex = sort_indices[i];


  float dens = vars_sorted[i+0*num].x;
  p /= params->simulation_scale;
  vars_unsorted[originalIndex+1 *num] = (float4)(p.xyz, dens);
  vars_unsorted[originalIndex+2 *num] = v;
  vars_unsorted[originalIndex+0 *num].x = vars_sorted[i+0*num].x;
  vars_unsorted[originalIndex+3 *num] = f;

  positions[originalIndex] = (float4)(p.xyz, 1.);
}
