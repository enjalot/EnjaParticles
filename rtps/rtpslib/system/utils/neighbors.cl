# 1 "neighbors.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "neighbors.cpp"



# 1 "cl_macros.h" 1
# 5 "neighbors.cpp" 2
# 1 "cl_structures.h" 1







typedef struct PointData
{
 float4 density;
 float4 color;
 float4 normal;
 float4 force;
 float4 surf_tens;
} PointData;

struct GridParams
{
    float4 grid_size;
    float4 grid_min;
    float4 grid_max;


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
    float alpha = -6.f*315.f/64.0f/params->PI/(h9*h9*h9);
    float Wij = alpha * hr2*hr2;
    return Wij;
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
    float alpha = 45.f/(params->PI * rlen*h6);
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
 pt->normal = (float4)(0.,0.,0.,0.);
 pt->force = (float4)(0.,0.,0.,0.);
 pt->surf_tens = (float4)(0.,0.,0.,0.);
}


void ForNeighbor(__global float4* vars_sorted,
    PointData* pt,
    __constant uint index_i,
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
# 21 "density_update.cl"
    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);
# 39 "density_update.cl"
 pt->density += (float4)(sphp->mass*Wij, 0., 0., 0.);
# 44 "neighbors.cpp" 2
 }

 if (fp->choice == 1) {

# 1 "pressure_update.cl" 1
# 16 "pressure_update.cl"
 float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);


 float4 di = vars_sorted[index_i+0*num].x;
 float4 dj = vars_sorted[index_j+0*num].x;


 float fact = 1.;



 float Pi = sphp->K*(di.x - 1000.f);
 float Pj = sphp->K*(dj.x - 1000.f);

 float kern = -dWijdr * (Pi + Pj)*0.5;
 float4 stress = kern*r;




 float4 veli = vars_sorted[index_i+2*num];
 float4 velj = vars_sorted[index_j+2*num];


 float vvisc = 1.000f;
 float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
 stress += vvisc * (velj-veli) * dWijlapl;
 stress *= sphp->mass/(di.x*dj.x);





 float Wijpol6 = Wpoly6(rlen, sphp->smoothing_distance, sphp);
 stress += (2.f * sphp->mass * (velj-veli)/(di.x+dj.x)
     * Wijpol6);


 pt->force += stress;
# 49 "neighbors.cpp" 2
 }

 if (fp->choice == 2) {

# 1 "normal_update.cl" 1




 float dWijdr = Wpoly6_dr(rlen, sphp->smoothing_distance, sphp);


 float4 dj = vars_sorted[index_j+0*num].x;

 pt->normal = -r * dWijdr * sphp->mass / dj.x;
# 54 "neighbors.cpp" 2
 }

 if (fp->choice == 3) {

 }
}

float4 ForPossibleNeighbor(__global float4* vars_sorted,
      PointData* pt,
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






 if (fp->choice == 0 || index_j != index_i) {


  float4 position_j = vars_sorted[index_j+1*num];


  float4 r = (position_i - position_j);

  float rlen = length(r);



  if (rlen <= sphp->smoothing_distance) {




   ForNeighbor(vars_sorted, pt, index_i, index_j, r, rlen, gp, fp, sphp , clf, cli);

  }
 }

}
