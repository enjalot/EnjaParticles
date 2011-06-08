#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

//float Wij = sphp->wpoly6_coef * Wpoly6(r, sphp->smoothing_distance, sphp);
float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);

pt->density.x += sphp->mass*Wij;
//pt->density.x += sphp->mass*Wij;
//----------------------------------------------------------------------
#endif
