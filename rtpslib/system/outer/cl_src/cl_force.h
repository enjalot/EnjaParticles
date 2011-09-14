/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#ifndef _PRESSURE_UPDATE_CL_
#define _PRESSURE_UPDATE_CL_

// gradient
// need to be careful, this kernel divides by rlen which could be 0
// once two particles assume the same position we will get a lot of branching
// and they won't split... how can we account for this?
//
// FIXED? I added 10E-6 to rlen during the division in Wspiky_dr kernel -IJ
// hacks, need to find the original cause (besides adding particles too fast)
/*
if(rlen == 0.0)
{
    rlen = 1.0;
    iej = 0;
}
*/
//this should 0 force between two particles if they get the same position
int rlencheck = rlen != 0.;
iej *= rlencheck;

float dWijdr = Wspiky_dr(rlen, sphp->smoothing_distance, sphp);

float4 di = density(index_i);  // should not repeat di=
float4 dj = density(index_j);
float idi = 1.0/di.x;
float idj = 1.0/dj.x;

//form simple SPH in Krog's thesis

float rest_density = 1000.f;
float Pi = sphp->K*(di.x - rest_density);
float Pj = sphp->K*(dj.x - rest_density);

float kern = -.5 * dWijdr * (Pi + Pj) * sphp->wspiky_d_coef;
//float kern = dWijdr * (Pi * idi * idi + Pj * idj * idj) * sphp->wspiky_d_coef;
float4 force = kern*r; 

float4 veli = veleval(index_i); // sorted
float4 velj = veleval(index_j);

#if 1
// Add viscous forces
float vvisc = sphp->viscosity;
float dWijlapl = Wvisc_lapl(rlen, sphp->smoothing_distance, sphp);
force += vvisc * (velj-veli) * dWijlapl;
#endif

force *=  sphp->mass/(di.x*dj.x);  // original
//force *=  sphp->mass;// /(di.x*dj.x); 

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
float4 xsph = (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di.x+dj.x));
pt->xsph += xsph * (float)iej;
pt->xsph.w = 0.f;
#endif

pt->force += force * (float)iej;


#endif
