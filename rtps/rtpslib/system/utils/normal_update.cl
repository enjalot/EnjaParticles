#ifndef _NORMAL_UPDATE_CL_
#define _NORMAL_UPDATE_CL_

	// gradient
	float dWijdr = Wpoly6_dr(rlen, sphp->smoothing_distance, sphp);

	//float4 di = density(index_i);  // should not repeat di=
	float4 dj = density(index_j);

	pt->normal = -r * dWijdr * sphp->mass / dj.x;

#endif
