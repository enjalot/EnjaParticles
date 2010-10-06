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

	float4 di = density(index_i);  // should not repeat di=
	float4 dj = density(index_j);

	//form simple SPH in Krog's thesis
	float Pi = sphp->K*(di.x - sphp->rest_density); 
	float Pj = sphp->K*(dj.x - sphp->rest_density);

	float kern = -0.5 * 1. * dWijdr * (Pi + Pj);
	float4 stress = kern*r;

	#if 1
	// Add viscous forces

	float4 veli = vel(index_i);
	float4 velj = vel(index_j);

	float vvisc = 0.001f; // SHOULD BE SET IN GE_SPH.cpp
	float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
	stress += vvisc * (velj-veli) * dWijlapl;
	stress *=  sphp->mass/(di.x*dj.x);  // original
	#endif


	#if 0
	// Add XSPH stabilization term
	float Wijpol6 = Wpoly6(rlen, sphp->smoothing_distance, sphp);
	//stress +=  (2.f * sphp->mass * (velj-veli)/(di.x+dj.x) 
    //    * Wijpol6) / sphp->dt;
	stress +=  (1.f * sphp->mass * (velj-veli)/(di.x+dj.x) 
	    * Wijpol6);
	#endif

	return stress;

#endif
