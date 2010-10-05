#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

#if 1
	float h = sphp->smoothing_distance;
	float re2 = h*h;
    float R = sqrt(rlen_sq/re2);
    float alpha = 315.f/208.f/sphp->PI/h/h/h;
	float Wij = alpha*(2.f/3.f - 9.f*R*R/8.f + 19.f*R*R*R/24.f - 5.f*R*R*R*R/32.f);
	int num = get_global_id(0); // for macro in next line
	density(index_i) += sphp->mass * Wij; 
	return sphp->mass*Wij;
#endif

	//return r;
//----------------------------------------------------------------------
#endif
