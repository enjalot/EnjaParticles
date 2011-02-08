#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

    //float Wij = flockp->wpoly6_coef * Wpoly6(r, flockp->smoothing_distance, flockp);
    float Wij = Wpoly6(r, flockp->smoothing_distance, flockp);

	pt->density.x += flockp->mass*Wij;
	//pt->density.x += flockp->mass*Wij;
//----------------------------------------------------------------------
#endif
