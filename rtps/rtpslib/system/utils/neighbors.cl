# 1 "neighbors.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "neighbors.cpp"



# 1 "cl_macros.h" 1
# 5 "neighbors.cpp" 2
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
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;
    float K;
};
# 6 "neighbors.cpp" 2
# 1 "wpoly6.cl" 1




float Wpoly6(float4 r, float h, __constant struct SPHParams* params)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
 float h9 = h*h;
 float hr2 = (h9-r2);
 h9 = h9*h;
    float alpha = 315.f/64.0f/params->PI/(h9*h9*h9);
    float Wij = alpha * hr2*hr2*hr2;
    return Wij;
}

float Wspiky(float rlen, float h, __constant struct SPHParams* params)
{
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;
 float hr2 = (h - rlen);
 float Wij = alpha * hr2*hr2*hr2/rlen;
 return Wij;
}
# 7 "neighbors.cpp" 2


float4 ForNeighbor(__global float4* vars_sorted,
    __constant uint index_i,
    uint index_j,
    float4 r,
    float rlen,
    float rlen_sq,
      __constant struct GridParams* gp,
      __constant struct FluidParams* fp,
      __constant struct SPHParams* sphp
      , __global float4* clf, __global int4* cli
    )
{
 int num = get_global_size(0);

 cli[index_i].x++;

 if (fp->choice == 0) {


# 1 "density_update.cl" 1





 float h = sphp->smoothing_distance;
 clf[index_i].x = h;
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
 float h9 = h*h;
 float hr2 = (h9-r2);
 h9 = h9*h;
    float alpha = 315.f/64.0f/sphp->PI/(h9*h9*h9);

 clf[index_i].z = hr2;
 clf[index_i].w = sphp->mass;
    float Wij = alpha * hr2*hr2*hr2;
# 35 "density_update.cl"
 return (float4)(sphp->mass*Wij, 0., 0., 0.);
# 29 "neighbors.cpp" 2
 }

 if (fp->choice == 1) {

# 1 "pressure_update.cl" 1



 float Wij = Wspiky(rlen, sphp->smoothing_distance, sphp);

 float di = vars_sorted[index_i+0*num].x;
 float dj = vars_sorted[index_j+0*num].x;


 float Pi = sphp->K*(di - 1000.0f);
 float Pj = sphp->K*(dj - 1000.0f);

 float kern = sphp->mass * 1.0f * Wij * (Pi + Pj) / (di * dj);





 return kern*r;
# 34 "neighbors.cpp" 2
 }
}

float4 ForPossibleNeighbor(__global float4* vars_sorted,
      __constant uint num,
      __constant uint index_i,
      uint index_j,
      __constant float4 position_i,
        __constant struct GridParams* gp,
        __constant struct FluidParams* fp,
        __constant struct SPHParams* sphp
        , __global float4* clf, __global int4* cli
      )
{
 float4 frce = (float4) (0.,0.,0.,0.);







 {

  float4 position_j = vars_sorted[index_j+1*num];


  float4 r = (position_i - position_j);

  float rlen_sq = dot(r,r);

  float rlen = length(r);




 clf[index_i].x = rlen;
 clf[index_i].y = sphp->smoothing_distance;
 return frce;

  if (rlen <= sphp->smoothing_distance) {
   cli[index_i].z++;

   frce = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, rlen_sq, gp, fp, sphp , clf, cli);

  }
 }
 return frce;
}
