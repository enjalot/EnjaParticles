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
    int numParticles;
};

struct FluidParams
{
 float smoothing_length;
 float scale_to_simulation;
 float mass;
 float dt;
 float friction_coef;
 float restitution_coef;
 float damping;
 float shear;
 float attraction;
 float spring;
 float gravity;
};
# 7 "uniform_hash.cpp" 2





int4 calcGridCell(float4 p, float4 grid_min, float4 grid_delta)
{




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
# 70 "uniform_hash.cpp"
    return (gz*grid_res.y + gy) * grid_res.x + gx;
}
# 89 "uniform_hash.cpp"
__kernel void hash(
           __global float4* dParticlePositions,
           __global uint* sort_hashes,
           __global uint* sort_indexes,
           __constant struct GridParams* gp)
{

    uint index = get_global_id(0);
    if (index >= gp->numParticles) return;


    float4 p = dParticlePositions[index];


    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_delta);
    bool wrap_edges = false;
    uint hash = (uint) calcGridHash(gridPos, gp->grid_res, wrap_edges);



    sort_hashes[index] = hash;
    int pp = (int) p.x;

    sort_indexes[index] = index;

}
