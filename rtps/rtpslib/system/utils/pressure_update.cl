#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

	// gradient
#if 1
	float h = sphp->smoothing_distance;
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/(sphp->PI * rlen*h6);
	float hr2 = (h - rlen);
	float dWijdr = -alpha * (hr2*hr2);
	clf[index_i].x = h; // hr2 == 0 WHY? 
	clf[index_i].y = rlen;
	clf[index_i].z = dWijdr;
	clf[index_i].z = hr2;
#else
	float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);
#endif

	float di = density(index_i);  // should not repeat di=
	float dj = density(index_j);

	//form simple SPH in Krog's thesis
	float Pi = sphp->K*(di - sphp->rest_density); 
	float Pj = sphp->K*(dj - sphp->rest_density);

	float kern = 0.5 * sphp->mass * dWijdr * (Pi + Pj) / (di * dj);

	//clf[index_i] += r*kern; // why is there a w component?
	//cli[index_i].w = 1;

	//clf[index_i] += Wij;
	//clf[index_i].w= Pi;  // Pi and Pj are negative!!
	//cli[index_i].y++;
	//cli[index_i].x = -998;

//clf[index_i].x = kern;
//clf[index_i].y = rlen;
//clf[index_i].w = -17.;

	return kern*r;  // original

	clf[index_i].x = sphp->smoothing_distance;
	clf[index_i].y = di;
	clf[index_i].z = Pi;
	clf[index_i].w = kern;   // last two are inf

#endif
