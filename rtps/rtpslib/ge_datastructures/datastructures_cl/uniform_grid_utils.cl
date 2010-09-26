# 1 "uniform_grid_utils.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "uniform_grid_utils.cpp"
# 19 "uniform_grid_utils.cpp"
# 1 "cl_macros.h" 1
# 20 "uniform_grid_utils.cpp" 2
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
# 21 "uniform_grid_utils.cpp" 2
# 1 "neighbors.cpp" 1







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
 int numParticles = gp->numParticles;
 float4 ri = vars_sorted[index_i+0*numParticles];
 float4 rj = vars_sorted[index_j+0*numParticles];
 float4 relPos = rj-ri;
 float dist = length(relPos);
 float collideDist = 2.*fp->smoothing_length;


 float4 force;
 force.x = 0.;
 force.y = 0.;
 force.z = 0.;
 force.w = 0.;

 if (dist < collideDist) {
  float4 vi = vars_sorted[index_i+1*numParticles];
  float4 vj = vars_sorted[index_j+1*numParticles];
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
      __constant uint numParticles,
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

  float4 position_j = vars_sorted[index_j+0*numParticles];


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
# 22 "uniform_grid_utils.cpp" 2




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
# 84 "uniform_grid_utils.cpp"
 return (gz*grid_res.y + gy) * grid_res.x + gx;
}




 float4 IterateParticlesInCell(
  __global float4* vars_sorted,
  __constant uint numParticles,
  __constant int4 cellPos,
  __constant uint index_i,
  __constant float4 position_i,
  __global int* cell_indexes_start,
  __global int* cell_indexes_end,
  __constant struct GridParams* gp,
  __constant struct FluidParams* fp
    )
 {



  uint cellHash = calcGridHash(cellPos, gp->grid_res, false);

  float4 force = convert_float4(0.0);


  uint startIndex = cell_indexes_start[cellHash];




  if (startIndex != 0xffffffff) {
   uint endIndex = cell_indexes_end[cellHash];


   for(uint index_j=startIndex; index_j < endIndex; index_j++) {



    force += ForPossibleNeighbor(vars_sorted, numParticles, index_i, index_j, position_i, gp, fp);

   }
  }
 }




 float4 IterateParticlesInNearbyCells(
  __global float4* vars_sorted,
  int numParticles,
  int index_i,
  __constant float4 position_i,
  __global int* cell_indices_start,
  __global int* cell_indices_end,
  __constant struct GridParams* gp,
  __constant struct FluidParams* fp)
 {


  float4 force;

  force.x = 0.;
  force.y = 0.;
  force.z = 0.;
  force.w = 0.;






  int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_delta);



  for(int z=cell.z-1; z<=cell.z+1; ++z) {
   for(int y=cell.y-1; y<=cell.y+1; ++y) {
    for(int x=cell.x-1; x<=cell.x+1; ++x) {
     int4 ipos;
     ipos.x = x;
     ipos.y = y;
     ipos.z = z;
     ipos.w = 1;


     force += IterateParticlesInCell(vars_sorted, numParticles, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp, fp);

    }
   }
  }

  return force;
 }





__kernel void K_SumStep1(
    uint numParticles,
    uint nb_vars,
    __global float4* vars,
    __global float4* sorted_vars,
          __global int* cell_indexes_start,
          __global int* cell_indexes_end,
    __constant struct GridParams* gp,
    __constant struct FluidParams* fp
    )
{

 int index = get_global_id(0);
    if (index >= numParticles) return;





    float4 position_i = sorted_vars[index+0*numParticles];



 float4 force;


    force = IterateParticlesInNearbyCells(sorted_vars, numParticles, index, position_i, cell_indexes_start, cell_indexes_end, gp, fp);
# 222 "uniform_grid_utils.cpp"
 vars[index+numParticles*2] = force;



}
