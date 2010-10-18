# 1 "neighbors.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "neighbors.cpp"



# 1 "cl_macros.h" 1
# 10 "cl_macros.h"
# 1 "../variable_labels.h" 1
# 11 "cl_macros.h" 2
# 5 "neighbors.cpp" 2
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


    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/(params->PI*rlen*h6);
 float hr2 = (h - rlen);
 float Wij = -alpha * (hr2*hr2);
 return Wij;
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
 float alpha = 15./(params->PI * h3*h3);
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

void ForNeighbor(__global float4* vars_sorted,
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
  cli[index_i].y++;



# 1 "density_update.cl" 1



    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);



 pt->density.x += sphp->mass*Wij;
# 45 "neighbors.cpp" 2
 }

 if (fp->choice == 1) {

# 1 "pressure_update.cl" 1




 float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

 float4 di = vars_sorted[index_i+0*num].x;
 float4 dj = vars_sorted[index_j+0*num].x;




 float rest_density = 1000.f;
 float Pi = sphp->K*(di.x - rest_density);
 float Pj = sphp->K*(dj.x - rest_density);

 clf[index_i].x = 45.;

 float kern = -dWijdr * (Pi + Pj)*0.5;

 float4 stress = kern*r;





 float4 veli = vars_sorted[index_i+2*num];
 float4 velj = vars_sorted[index_j+2*num];




 float vvisc = 0.001f;

 float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
 stress += vvisc * (velj-veli) * dWijlapl;


 stress *= sphp->mass/(di.x*dj.x);



 float Wijpol6 = Wpoly6(rlen, sphp->smoothing_distance, sphp);
 pt->xsph += (2.f * sphp->mass * (velj-veli)/(di.x+dj.x) * Wijpol6);
 pt->xsph.w = 0.f;


 pt->force += stress;
# 50 "neighbors.cpp" 2
 }

 if (fp->choice == 2) {

# 1 "surface_tension_update.cl" 1




 float dWijdr = Wpoly6_dr(rlen, sphp->smoothing_distance, sphp);





 float4 dj = vars_sorted[index_j+0*num].x;
 pt->color_normal += -r * dWijdr * sphp->mass / dj.x;


 float dWijlapl = Wpoly6_lapl(rlen, sphp->smoothing_distance, sphp);
 pt->color_lapl += -sphp->mass * dWijlapl / dj.x;
# 55 "neighbors.cpp" 2
 }

 if (fp->choice == 3) {

 }
}

void ForPossibleNeighbor(__global float4* vars_sorted,
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
# 89 "neighbors.cpp"
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
