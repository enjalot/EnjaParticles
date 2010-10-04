#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

#if -1
	float Wij;
	#if 1
	float pi45 = 45.f/sphp->PI;
	float h = sphp->smoothing_distance;
    float h3 = h*h*h;
    float alpha = pi45/h3;
	float hr2 = 1.f - rlen/h;
	Wij = alpha * hr2*hr2*hr2/rlen;
	//clf[index_i].x = h;
	//clf[index_i].y = h3;
	//clf[index_i].z = Wij;
	//clf[index_i].w = alpha;
	#endif
	//return r;
#else
	float Wij = Wspiky(rlen, sphp->smoothing_distance, sphp);
#endif

	float di = density(index_i);  // should not repeat di=
	float dj = density(index_j);

	//form simple SPH in Krog's thesis
	float Pi = sphp->K*(di - sphp->rest_density); 
	float Pj = sphp->K*(dj - sphp->rest_density);

	float kern = sphp->mass * 1.0f * Wij * (Pi + Pj) / (di * dj);

	//clf[index_i] += r*kern; // why is there a w component?
	cli[index_i].w = 1;

	//clf[index_i] += Wij;
	//clf[index_i].w= Pi;  // Pi and Pj are negative!!
	//cli[index_i].y++;
	//cli[index_i].x = -998;

	return kern*r;  // original

	clf[index_i].x = sphp->smoothing_distance;
	clf[index_i].y = di;
	clf[index_i].z = Pi;
	clf[index_i].w = kern;   // last two are inf
	return Wij;   // debugging

#endif
