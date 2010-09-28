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
};
# 6 "neighbors.cpp" 2


float4 ForNeighbor(__global float4* vars_sorted,
    __constant uint index_i,
    uint index_j,
    float4 r,
    float rlen,
    float rlen_sq,
      __constant struct GridParams* gp,
      __constant struct FluidParams* fp)
{




 int index = get_global_id(0);

# 1 "cl_snippet_sphere_forces.h" 1
# 18 "cl_snippet_sphere_forces.h"
 int num = gp->num;
 float4 ri = vars_sorted[index_i+1*num];
 float4 rj = vars_sorted[index_j+1*num];
 float4 relPos = rj-ri;
 float dist = length(relPos);
 float collideDist = 2.*fp->smoothing_length;


 float4 force;
 force.x = 0.;
 force.y = 0.;
 force.z = 0.;
 force.w = 0.;

 if (dist < collideDist) {
  float4 vi = vars_sorted[index_i+2*num];
  float4 vj = vars_sorted[index_j+2*num];
  float4 norm = relPos / dist;


  float4 relVel = vj - vi;


  float4 tanVel = relVel - (dot(relVel, norm) * norm);


  force = -fp->spring*(collideDist - dist) * norm;


  force +=fp->damping*relVel;


  force += fp->shear*tanVel;
  force += fp->attraction*relPos;
 }
# 24 "neighbors.cpp" 2

 return force;
}

float4 ForPossibleNeighbor(__global float4* vars_sorted,
      __constant uint num,
      __constant uint index_i,
      uint index_j,
      __constant float4 position_i,
        __constant struct GridParams* gp,
        __constant struct FluidParams* fp)
{
 float4 force;
 force.x = 0.;
 force.y = 0.;
 force.z = 0.;
 force.w = 0.;



 if (index_j != index_i) {

  float4 position_j = vars_sorted[index_j+1*num];


  float4 r = (position_i - position_j) * fp->scale_to_simulation;

  float rlen_sq = dot(r,r);

  float rlen;
  rlen = sqrt(rlen_sq);


  if (rlen <= fp->smoothing_length) {

   force = ForNeighbor(vars_sorted, index_i, index_j, r, rlen, rlen_sq, gp, fp);

  }
 }
 return force;
}
