# 1 "collision_wall_cl.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "collision_wall_cl.cpp"



# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 5 "collision_wall_cl.cpp" 2
# 1 "cl_structures.h" 1






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
# 6 "collision_wall_cl.cpp" 2




float4 calculateRepulsionForce(
      float4 normal,
   float4 vel,
   float boundary_stiffness,
   float boundary_dampening,
   float boundary_distance)
{





    vel.w = 0.0f;
    float4 repulsion_force = (boundary_stiffness * boundary_distance - boundary_dampening * dot(normal, vel))*normal;
 repulsion_force.w = 0.f;
    return repulsion_force;
}



float4 calculateFrictionForce(float4 vel, float4 force, float4 normal, float friction_kinetic, float friction_static_limit)
{
 float4 friction_force = (float4)(0.0f,0.0f,0.0f,0.0f);
    force.w = 0.0f;
    vel.w = 0.0f;


 float4 f_n = force * dot(normal, force);

 float4 f_t = force - f_n;


 float4 v_n = vel * dot(normal, vel);

 float4 v_t = vel - v_n;

 if((v_t.x + v_t.y + v_t.z)/3.0f > friction_static_limit)
  friction_force = -v_t;
 else
  friction_force = friction_kinetic * -v_t;
# 59 "collision_wall_cl.cpp"
 return friction_force;
}


__kernel void collision_wall(
  __global float4* vars_sorted,
  __constant struct GridParams* gp,
  __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);
 int num = get_global_size(0);
 int nb_vars = gp->nb_vars;

    float4 p = vars_sorted[i+1*num];
    float4 v = vars_sorted[i+8*num];
 float4 f = vars_sorted[i+3*num];
    float4 r_f = (float4)(0.f, 0.f, 0.f, 0.f);
    float4 f_f = (float4)(0.f, 0.f, 0.f, 0.f);



    float friction_kinetic = 0.0f;
    float friction_static_limit = 0.0f;


    float diff = params->boundary_distance - (p.z - gp->bnd_min.z);
    if (diff > params->EPSILON)
    {

        float4 normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }


    diff = params->boundary_distance - (gp->bnd_max.z - p.z);
    if (diff > params->EPSILON)
    {

        float4 normal = (float4)(0.0f, 0.0f, -1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }


    diff = params->boundary_distance - (p.y - gp->bnd_min.y);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }
    diff = params->boundary_distance - (gp->bnd_max.y - p.y);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, -1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }


    diff = params->boundary_distance - (p.x - gp->bnd_min.x);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }
    diff = params->boundary_distance - (gp->bnd_max.x - p.x);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(-1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }



 vars_sorted[i+3*num] += r_f + f_f;
}
