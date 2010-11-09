#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

	// gradient
	float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

	float4 di = density(index_i);  // should not repeat di=
	float4 dj = density(index_j);

	//form simple SPH in Krog's thesis

	float rest_density = 1000.f;
	float Pi = sphp->K*(di.x - rest_density);
	float Pj = sphp->K*(dj.x - rest_density);

	float kern = -dWijdr * (Pi + Pj)*0.5 * sphp->wspiky_d_coef;
	float4 stress = kern*r; // correct version

	float4 veli = veleval(index_i); // sorted
	float4 velj = veleval(index_j);

	// Add viscous forces

	#if 1
	float vvisc = .001f; // SHOULD BE SET IN GE_SPH.cpp
	float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
	stress += vvisc * (velj-veli) * dWijlapl;
	#endif

	stress *=  sphp->mass/(di.x*dj.x);  // original

	#if 1
	// Add XSPH stabilization term
    // the poly6 kernel calculation seems to be wrong, using rlen as a vector when it is a float...
	//float Wijpol6 = Wpoly6(r, sphp->smoothing_distance, sphp) * sphp->wpoly6_coeff;
    /*
    float h = sphp->smoothing_distance;
    float hr2 = (h*h - rlen*rlen);
	float Wijpol6 = hr2*hr2*hr2;// * sphp->wpoly6_coeff;
    */
	float Wijpol6 = Wpoly6(r, sphp->smoothing_distance, sphp);
	//float Wijpol6 = sphp->wpoly6_coef * Wpoly6(rlen, sphp->smoothing_distance, sphp);
	pt->xsph +=  (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di.x+dj.x));
	pt->xsph.w = 0.f;
	#endif

	pt->force += stress;

#endif
