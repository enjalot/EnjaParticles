# 1 "uniform_hash.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "uniform_hash.cpp"





# 1 "cl_structures.h" 1






struct GridParams
{
    float4 grid_size;
    float4 grid_min;
    float4 grid_max;


    float4 grid_res;
    float4 grid_delta;
    float4 grid_inv_delta;
    int numParticles;
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
# 7 "uniform_hash.cpp" 2
# 1 "cl_macros.h" 1
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


uint calcGridHash(int4 gridPos, float4 grid_res, __constant bool wrapEdges)
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







    return (gz*grid_res.y + gy) * grid_res.x + gx;
}
# 89 "uniform_hash.cpp"
__kernel void hash(

           __global float4* vars_unsorted,
           __global uint* sort_hashes,
           __global uint* sort_indexes,
           __global uint* cell_indices_start,
           __constant struct GridParams* gp)


{

    uint index = get_global_id(0);

 int num = get_global_size(0);
    if (index >= num) return;


 cell_indices_start = 0xffffffff;



    float4 p = vars_unsorted[index+1*num];


    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_inv_delta);
    bool wrap_edges = false;
    uint hash = (uint) calcGridHash(gridPos, gp->grid_res, wrap_edges);




    sort_hashes[index] = hash;
    int pp = (int) p.x;

    sort_indexes[index] = index;


}
