#ifndef _WPOLY6_CPU_
#define _WPOLY6_CPU_

#include "../GE_SPH.h"
#include "structs.h"

namespace rtps {

//----------------------------------------------------------------------
float Wpoly6(float4 r, float h,  struct GE_SPHParams* params)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
	float h3 = h*h;
	float hr2 = (h3-r2); // h9 = h^2
	//printf("hr2= %f\n", hr2);
	h3 = h3*h;   //  h3 = h^3
    float alpha = 315.f/64.0f/params->PI/(h3*h3*h3);
	//float alpha = 0.f;
    float Wij = alpha * hr2*hr2*hr2;
	//Wij = 0.f;
	//printf("Wij,hr2,alpha= %f, %f, %f, h^3= %f, h= %\n", Wij, hr2, alpha, h3, h);
    return Wij;
}
//----------------------------------------------------------------------
float Wpoly6_dr(float4 r, float h,  struct GE_SPHParams* params)
{
// Derivative with respect to |r| divided by |r|
//   W_{|r|}/r = -2*(315/64*pi*h^9) 3 (h^2-r^2)^2 
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
	float h9 = h*h;
	float hr2 = (h9-r2); // h9 = h^2
	h9 = h9*h;   //  h9 = h^3
    float alpha = -945.f/(32.0f*params->PI*h9*h9*h9);
    float Wij = alpha * hr2*hr2;
    return Wij;
}
//----------------------------------------------------------------------
float Wpoly6_lapl(float4 r, float h,  struct GE_SPHParams* params)
{
// Laplacian
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
	float h2 = h*h;
	float h3 = h2*h;
	float alpha = -945.f/(32.0f*params->PI*h3*h3*h3);
	float Wij = alpha*(h2-r2)*(2.*h2-7.f*r2);
}
//----------------------------------------------------------------------
float Wspiky(float rlen, float h,  struct GE_SPHParams* params)
{
    float h6 = h*h*h * h*h*h;
    float alpha = 15.f/(params->PI*h6);
	float hr2 = (h - rlen);
	float Wij = alpha * hr2*hr2*hr2;
	return Wij;
}
//----------------------------------------------------------------------
float Wspiky_dr(float rlen, float h,  struct GE_SPHParams* params)
{
// Derivative with respect to |r| divided by |r|
//   W_{|r|}/r = (45/pi*h^6) (h-|r|)^2 (-1) / r
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/(params->PI*rlen*h6);
	float hr2 = (h - rlen);
	float Wij = -alpha * (hr2*hr2);
	return Wij;
}
//----------------------------------------------------------------------
float Wvisc(float rlen, float h,  struct GE_SPHParams* params)
{
	float alpha = 15./(2.*params->PI*h*h*h);
	float rh = rlen / h;
	float Wij = rh*rh*(1.-0.5*rh) + 0.5/rh - 1.;
	return alpha*Wij;
}
//----------------------------------------------------------------------
float Wvisc_dr(float rlen, float h,  struct GE_SPHParams* params)
// Derivative with respect to |r| divided by |r|
// 
{
	float alpha = 15./(2.*params->PI * h*h*h);
	float rh = rlen / h;
	float Wij = (-1.5*rh + 2.)/(h*h) - 0.5/(rh*rlen*rlen);
	return Wij;
}
//----------------------------------------------------------------------
float Wvisc_lapl(float rlen, float h,  struct GE_SPHParams* params)
{
	float h3 = h*h*h;
	float alpha = 15./(params->PI * h3*h3);
	float Wij = alpha*(h-rlen);
	return Wij;
}
//----------------------------------------------------------------------

} // namespace

#endif // _WPOLY6_CL_

