# 1 "uniform_hash.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "uniform_hash.cpp"





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
 int4 expo;
 int4 shift[27];
    int numParticles;
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
 int4 expo;
 int4 shift[27];
    int numParticles;
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
# 7 "uniform_hash.cpp" 2
# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 8 "uniform_hash.cpp" 2





int4 calcGridCell(float4 p, float4 grid_min, float4 grid_inv_delta)
{




    float4 pp;
    pp.x = (p.x-grid_min.x)*grid_inv_delta.x;
    pp.y = (p.y-grid_min.y)*grid_inv_delta.y;
    pp.z = (p.z-grid_min.z)*grid_inv_delta.z;
    pp.w = (p.w-grid_min.w)*grid_inv_delta.w;

    int4 ii;
    ii.x = (int) pp.x;
    ii.y = (int) pp.y;
    ii.z = (int) pp.z;
    ii.w = (int) pp.w;
    return ii;
}


int calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges
           , __global float4* fdebug,
           __global int4* idebug
   )
{

    int gx;
    int gy;
    int gz;

    if(wrapEdges) {
        int gsx = (int)floor(grid_res.x);
        int gsy = (int)floor(grid_res.y);
        int gsz = (int)floor(grid_res.z);







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







 int index = get_global_id(0);
# 84 "uniform_hash.cpp"
    return (gz*grid_res.y + gy) * grid_res.x + gx;
}
# 103 "uniform_hash.cpp"
__kernel void hash(
           __global float4* vars_unsorted,
           __global int* sort_hashes,
           __global int* sort_indexes,

           __constant struct GridParams* gp,
           __global float4* fdebug,
           __global int4* idebug
     )
{


    int index = get_global_id(0);

 int num = get_global_size(0);
    if (index >= num) return;
# 127 "uniform_hash.cpp"
    float4 p = vars_unsorted[index+1 *num];


    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_inv_delta);
    bool wrap_edges = false;
    int hash = (int) calcGridHash(gridPos, gp->grid_res, wrap_edges, fdebug, idebug);



    sort_hashes[index] = hash;
    int pp = (int) p.x;

    sort_indexes[index] = index;



}
