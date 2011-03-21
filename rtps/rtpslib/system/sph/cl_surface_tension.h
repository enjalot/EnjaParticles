#ifndef _NORMAL_UPDATE_CL_
#define _NORMAL_UPDATE_CL_

// gradient
float dWijdr = Wpoly6_dr(rlen, sphp->smoothing_distance, sphp);

// CHECK that r has no w component !!!

// uses color which is 1 everywhere
// mass/rho = estimate of volume element 
float4 dj = density(index_j);
pt->color_normal += -r * dWijdr * sphp->mass / dj.x;


float dWijlapl = Wpoly6_lapl(rlen, sphp->smoothing_distance, sphp);
pt->color_lapl += -sphp->mass * dWijlapl / dj.x;

#endif
