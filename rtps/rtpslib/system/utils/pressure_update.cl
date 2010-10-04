#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

	float Wij = Wspiky(rlen, sphp->smoothing_distance, sphp);

	float di = density(index_i);  // should not repeat di=
	float dj = density(index_j);

	//form simple SPH in Krog's thesis
	float Pi = sphp->K*(di - sphp->rest_density); 
	float Pj = sphp->K*(dj - sphp->rest_density);

	float kern = sphp->mass * Wij * (Pi + Pj) / (di * dj);

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
