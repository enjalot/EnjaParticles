#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);
    //float Wij = Wpoly6(rlen, fp->smoothing_length, sphp);

	#if 0
	clf[index_i].x = Wij; // Wij too small
	clf[index_i].y = rlen;
	// what is sphp->smoothing_distance?
	//clf[index_i].y = sphp->smoothing_distance; // smoothing_distance = 0.05 ... (way too small)
	clf[index_i].z = fp->smoothing_length; // smoothing_distance = 0.05 ... (way too small)
	clf[index_i].w = sphp->smoothing_distance; // smoothing_distance = 0.05 ... (way too small)
	clf[index_i] = r;
	#endif

	//clf[index_i].x = sphp->mass;
	//clf[index_i].y = Wij;
	//cli[index_i].x = -17.;

	pt->density += (float4)(sphp->mass*Wij, 0., 0., 0.);
	//return (float4)(sphp->mass*Wij, 0., 0., 0.);
//----------------------------------------------------------------------
#endif
