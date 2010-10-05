#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

//=====
	#if 0
	float h = sphp->smoothing_distance;
	clf[index_i].x = h;
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
	float h9 = h*h;
	float hr2 = (h9-r2); // h9 = h^2
	h9 = h9*h;   //  h9 = h^3
    float alpha = 315.f/64.0f/sphp->PI/(h9*h9*h9);
	//clf[index_i].y = alpha;
	clf[index_i].z = hr2;
	clf[index_i].w = sphp->mass;
    float Wij = alpha * hr2*hr2*hr2;
	#endif
//=====

#if 1
    float Wij = Wpoly6(r, sphp->smoothing_distance, sphp);
    //float Wij = Wpoly6(rlen, fp->smoothing_length, sphp);
#endif

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
	return (float4)(sphp->mass*Wij, 0., 0., 0.);
	//return (float)(1., 0., 0., 0.);  // FOR DEBUGGING
//----------------------------------------------------------------------
#endif
