#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

	// gradient
	float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

	float4 di = density(index_i);  // should not repeat di=
	float4 dj = density(index_j);

	//form simple SPH in Krog's thesis
	float fact = 1.; // 5.812
	// rest density does not appear to be correct. 
	//float Pi = sphp->K*(di.x - fact * sphp->rest_density); 
	//float Pj = sphp->K*(dj.x - fact * sphp->rest_density);

	//float rest_density = 00.f;
	float rest_density = 1000.f;
	float Pi = sphp->K*(di.x - rest_density);
	float Pj = sphp->K*(dj.x - rest_density);

	float kern = -dWijdr * (Pi + Pj)*0.5;
	float4 stress = kern*r;

	float4 veli = vel(index_i);
	float4 velj = vel(index_j);

	#if 1
	// Add viscous forces

	float vvisc = 0.0001f; // SHOULD BE SET IN GE_SPH.cpp
	//float vvisc = 1.000f; // SHOULD BE SET IN GE_SPH.cpp
	float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
	stress += vvisc * (velj-veli) * dWijlapl;
	#endif

	stress *=  sphp->mass/(di.x*dj.x);  // original


	#if 1
	// Add XSPH stabilization term
	float Wijpol6 = Wpoly6(rlen, sphp->smoothing_distance, sphp);
	float4 surf_tens =  (2.f * sphp->mass * (velj-veli)/(di.x+dj.x) 
	    * Wijpol6);
	//surf_tens.w = 0.f;
	stress += surf_tens;
	#endif

	pt->force += stress;


	//return stress;

#endif
