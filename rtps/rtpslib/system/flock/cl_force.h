#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

	// gradient
	float dWijdr = Wspiky_dr(rlen, flockp->smoothing_distance, flockp);

	float4 di = density(index_i);  // should not repeat di=
	float4 dj = density(index_j);

	//form simple FLOCK in Krog's thesis

	float rest_density = 1000.f;
	float Pi = flockp->K*(di.x - rest_density);
	float Pj = flockp->K*(dj.x - rest_density);

	float kern = -dWijdr * (Pi + Pj)*0.5 * flockp->wspiky_d_coef;
	float4 stress = kern*r; // correct version

	float4 veli = veleval(index_i); // sorted
	float4 velj = veleval(index_j);

	// Add viscous forces

	#if 1
	float vvisc = flockp->viscosity;
	float dWijlapl = Wvisc_lapl(rlen, flockp->smoothing_distance, flockp);
	stress += vvisc * (velj-veli) * dWijlapl;
	#endif

	stress *=  flockp->mass/(di.x*dj.x);  // original

	#if 1
	// Add XFLOCK stabilization term
    // the poly6 kernel calculation seems to be wrong, using rlen as a vector when it is a float...
	//float Wijpol6 = Wpoly6(r, flockp->smoothing_distance, flockp) * flockp->wpoly6_coeff;
    /*
    float h = flockp->smoothing_distance;
    float hr2 = (h*h - rlen*rlen);
	float Wijpol6 = hr2*hr2*hr2;// * flockp->wpoly6_coeff;
    */
	float Wijpol6 = Wpoly6(r, flockp->smoothing_distance, flockp);
	//float Wijpol6 = flockp->wpoly6_coef * Wpoly6(rlen, flockp->smoothing_distance, flockp);
	pt->xflock +=  (2.f * flockp->mass * Wijpol6 * (velj-veli)/(di.x+dj.x));
	pt->xflock.w = 0.f;
	#endif

	pt->force += stress;

#endif
