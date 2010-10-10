#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);

	//pt->density += (float4)(sphp->mass*Wij, 0., 0., 0.);

	pt->density.x += sphp->mass*Wij;
//----------------------------------------------------------------------
#endif
