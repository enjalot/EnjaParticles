#ifndef _DENSITY_DENOM_UPDATE_CL_
#define _DENSITY_DENOM_UPDATE_CL_

	//cli[index_i].z = -75;
	//cli[index_i].y = fp->choice;

    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);

	pt->density.y += sphp->mass*Wij / density(index_i);
//----------------------------------------------------------------------
#endif
