# 1 "uniform_grid_utils.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "uniform_grid_utils.cpp"
# 12 "uniform_grid_utils.cpp"
# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 13 "uniform_grid_utils.cpp" 2
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
 int4 expo;
 int4 shift[27];
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
# 14 "uniform_grid_utils.cpp" 2
# 1 "neighbors.cpp" 1





# 1 "wpoly6.cl" 1




float Wpoly6_glob(float4 r, float h)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;

 float hr2 = (h*h-r2);

 if (hr2 > 0) {
  return hr2*hr2*hr2;
 } else {
  return 0.;
 }
}

float Wpoly6(float4 r, float h, __constant struct SPHParams* params)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
# 29 "wpoly6.cl"
 float hr2 = (h*h-r2);

 return hr2*hr2*hr2;


}

float Wpoly6_dr(float4 r, float h, __constant struct SPHParams* params)
{


    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
 float h9 = h*h;
 float hr2 = (h9-r2);
 h9 = h9*h;
    float alpha = -945.f/(32.0f*params->PI*h9*h9*h9);
    float Wij = alpha * hr2*hr2;
    return Wij;
}

float Wpoly6_lapl(float4 r, float h, __constant struct SPHParams* params)
{

    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
 float h2 = h*h;
 float h3 = h2*h;
 float alpha = -945.f/(32.0f*params->PI*h3*h3*h3);
 float Wij = alpha*(h2-r2)*(2.*h2-7.f*r2);
}

float Wspiky(float rlen, float h, __constant struct SPHParams* params)
{
    float h6 = h*h*h * h*h*h;
    float alpha = 15.f/params->PI/h6;
 float hr2 = (h - rlen);
 float Wij = alpha * hr2*hr2*hr2;
 return Wij;
}

float Wspiky_dr(float rlen, float h, __constant struct SPHParams* params)
{
# 79 "wpoly6.cl"
 float hr2 = h - rlen;
 return -hr2*hr2/rlen;

}

float Wvisc(float rlen, float h, __constant struct SPHParams* params)
{
 float alpha = 15./(2.*params->PI*h*h*h);
 float rh = rlen / h;
 float Wij = rh*rh*(1.-0.5*rh) + 0.5/rh - 1.;
 return alpha*Wij;
}

float Wvisc_dr(float rlen, float h, __constant struct SPHParams* params)


{
 float alpha = 15./(2.*params->PI * h*h*h);
 float rh = rlen / h;
 float Wij = (-1.5*rh + 2.)/(h*h) - 0.5/(rh*rlen*rlen);
 return Wij;
}

float Wvisc_lapl(float rlen, float h, __constant struct SPHParams* params)
{
 float h3 = h*h*h;
 float alpha = 45./(params->PI * h3*h3);
 float Wij = alpha*(h-rlen);
 return Wij;
}
# 7 "neighbors.cpp" 2


void zeroPoint(PointData* pt)
{
 pt->density = (float4)(0.,0.,0.,0.);
 pt->color = (float4)(0.,0.,0.,0.);
 pt->color_normal = (float4)(0.,0.,0.,0.);
 pt->force = (float4)(0.,0.,0.,0.);
 pt->surf_tens = (float4)(0.,0.,0.,0.);
 pt->color_lapl = 0.;
 pt->xsph = (float4)(0.,0.,0.,0.);
}

inline void ForNeighbor(__global float4* vars_sorted,
    PointData* pt,
    uint index_i,
    uint index_j,
    float4 r,
    float rlen,
      __constant struct GridParams* gp,
      __constant struct FluidParams* fp,
      __constant struct SPHParams* sphp
      , __global float4* clf, __global int4* cli
    )
{
 int num = get_global_size(0);


 if (fp->choice == 0) {


# 1 "density_update.cl" 1




    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);

 pt->density.x += sphp->mass*Wij;
# 39 "neighbors.cpp" 2
 }

 if (fp->choice == 1) {

# 1 "pressure_update.cl" 1




 float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

 float4 di = vars_sorted[index_i+0*num].x;
 float4 dj = vars_sorted[index_j+0*num].x;



 float rest_density = 1000.f;
 float Pi = sphp->K*(di.x - rest_density);
 float Pj = sphp->K*(dj.x - rest_density);

 float kern = -dWijdr * (Pi + Pj)*0.5 * sphp->wspike_d_coef;
 float4 stress = kern*r;

 float4 veli = vars_sorted[index_i+8*num];
 float4 velj = vars_sorted[index_j+8*num];
# 30 "pressure_update.cl"
 stress *= sphp->mass/(di.x*dj.x);



 float Wijpol6 = Wpoly6(rlen, sphp->smoothing_distance, sphp);

 pt->xsph += (2.f * sphp->mass * (velj-veli)/(di.x+dj.x) * Wijpol6);
 pt->xsph.w = 0.f;


 pt->force += stress;
# 44 "neighbors.cpp" 2
 }

 if (fp->choice == 2) {

# 1 "surface_tension_update.cl" 1




 float dWijdr = Wpoly6_dr(rlen, sphp->smoothing_distance, sphp);





 float4 dj = vars_sorted[index_j+0*num].x;
 pt->color_normal += -r * dWijdr * sphp->mass / dj.x;


 float dWijlapl = Wpoly6_lapl(rlen, sphp->smoothing_distance, sphp);
 pt->color_lapl += -sphp->mass * dWijlapl / dj.x;
# 49 "neighbors.cpp" 2
 }

 if (fp->choice == 3) {

 }
}

inline void ForPossibleNeighbor(__global float4* vars_sorted,
      PointData* pt,
      uint num,
      uint index_i,
      uint index_j,
      float4 position_i,
        __constant struct GridParams* gp,
        __constant struct FluidParams* fp,
        __constant struct SPHParams* sphp
        , __global float4* clf, __global int4* cli
      )
{
# 77 "neighbors.cpp"
 if (fp->choice == 0 || (index_j != index_i)) {


  float4 position_j = vars_sorted[index_j+1*num];


  float4 r = (position_i - position_j);
  r.w = 0.f;

  float rlen = length(r);



  if (rlen <= sphp->smoothing_distance) {


   ForNeighbor(vars_sorted, pt, index_i, index_j, r, rlen, gp, fp, sphp , clf, cli);

  }
 }
}
# 15 "uniform_grid_utils.cpp" 2




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


uint calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges)
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




 void IterateParticlesInCell(
  __global float4* vars_sorted,
  PointData* pt,
  uint num,
  int4 cellPos,
  uint index_i,
  float4 position_i,
  __global int* cell_indexes_start,
  __global int* cell_indexes_end,
  __constant struct GridParams* gp,
  __constant struct FluidParams* fp,
  __constant struct SPHParams* sphp
  , __global float4* clf, __global int4* cli
    )
 {

  uint cellHash = calcGridHash(cellPos, gp->grid_res, false);


  uint startIndex = cell_indexes_start[cellHash];



  if (startIndex != 0xffffffff) {
   uint endIndex = cell_indexes_end[cellHash];


   for(uint index_j=startIndex; index_j < endIndex; index_j++) {


    ForPossibleNeighbor(vars_sorted, pt, num, index_i, index_j, position_i, gp, fp, sphp , clf, cli);

   }
  }
 }




 void IterateParticlesInNearbyCells(
  __global float4* vars_sorted,
  PointData* pt,
  int num,
  int index_i,
  float4 position_i,
  __global int* cell_indices_start,
  __global int* cell_indices_end,
  __constant struct GridParams* gp,
  __constant struct FluidParams* fp,
  __constant struct SPHParams* sphp
  , __global float4* clf, __global int4* cli
  )
 {



  int4 cell = calcGridCell(position_i, gp->grid_min, gp->grid_inv_delta);



  for(int z=cell.z-1; z<=cell.z+1; ++z) {
   for(int y=cell.y-1; y<=cell.y+1; ++y) {
    for(int x=cell.x-1; x<=cell.x+1; ++x) {
     int4 ipos = (int4) (x,y,z,1);


     IterateParticlesInCell(vars_sorted, pt, num, ipos, index_i, position_i, cell_indices_start, cell_indices_end, gp, fp, sphp , clf, cli);



    }
   }
  }
 }





__kernel void K_SumStep1(
    __global float4* vars_sorted,
    __global int4* cell_compact,
    __constant struct GPUReturnValues* rv,
          __global int* cell_indexes_start,
          __global int* cell_indexes_end,
    __constant struct GridParams* gp,
    __constant struct FluidParams* fp,
    __constant struct SPHParams* sphp
    , __global float4* clf, __global int4* cli
    )
{

 int nb_vars = gp->nb_vars;
 int num = gp->num;


 int index = get_global_id(0);
    if (index >= num) return;




    float4 position_i = vars_sorted[index+1*num];




 PointData pt;
 zeroPoint(&pt);

 if (fp->choice == 0) {
     IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, fp, sphp , clf, cli);
  vars_sorted[index+0*num].x = sphp->wpoly6_coef * pt.density.x;

 }
 if (fp->choice == 1) {
     IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, fp, sphp , clf, cli);
  vars_sorted[index+3*num] = pt.force;
  vars_sorted[index+9*num] = sphp->wpoly6_coef * pt.xsph;

 }
 if (fp->choice == 2) {
     IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, fp, sphp , clf, cli);
  float norml = length(pt.color_normal);
  if (norml > 1.) {
   float4 stension = -0.3f * pt.color_lapl * pt.color_normal / norml;
   vars_sorted[index+3*num] += stension;
  }
 }
 if (fp->choice == 3) {
     IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, fp, sphp , clf, cli);


  vars_sorted[index+0*num].x /= pt.density.y;
 }
}
