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


#ifndef _WPOLY6_CL_
#define _WPOLY6_CL_

//----------------------------------------------------------------------
float Wpoly6(float4 r, float h, __constant struct SPHParams* params)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);

#if 0
    float h9 = h*h;
    float hr2 = (h9-r2); // h9 = h^2
    h9 = h9*h;   //  h9 = h^3
    float alpha = 315.f/64.0f/params->PI/(h9*h9*h9);
    return alpha * hr2*hr2*hr2;
#else 
    float hr2 = (h*h-r2); 
    //return params->wpoly6_coef * hr2*hr2*hr2;
    return hr2*hr2*hr2;
#endif

}
//----------------------------------------------------------------------
float Wpoly6_dr(float4 r, float h, __constant struct SPHParams* params)
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
float Wpoly6_lapl(float4 r, float h, __constant struct SPHParams* params)
{
    // Laplacian
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
    float h2 = h*h;
    float h3 = h2*h;
    float alpha = -945.f/(32.0f*params->PI*h3*h3*h3);
    float Wij = alpha*(h2-r2)*(2.*h2-7.f*r2);
    return Wij;
}
//----------------------------------------------------------------------
float Wspiky(float rlen, float h, __constant struct SPHParams* params)
{
    float h6 = h*h*h * h*h*h;
    float alpha = 15.f/params->PI/h6;
    float hr2 = (h - rlen);
    float Wij = alpha * hr2*hr2*hr2;
    return Wij;
}

//----------------------------------------------------------------------
float Wspiky_dr(float rlen, float h, __constant struct SPHParams* params)
{
    // Derivative with respect to |r| divided by |r|
    //   W_{|r|}/r = (45/pi*h^6) (h-|r|)^2 (-1) / r
#if 0
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/(params->PI*rlen*h6);
    float hr2 = (h - rlen);
    float Wij = -alpha * (hr2*hr2);
    return Wij;
#else
    float hr2 = h - rlen;
    //return -hr2*hr2/rlen;
    //return hr2*hr2/(rlen + params->EPSILON);
    return hr2*hr2/rlen;
#endif
}

//----------------------------------------------------------------------
float Wvisc(float rlen, float h, __constant struct SPHParams* params)
{
    float alpha = 15./(2.*params->PI*h*h*h);
    float rh = rlen / h;
    float Wij = rh*rh*(1.-0.5*rh) + 0.5/rh - 1.;
    return alpha*Wij;
}
//----------------------------------------------------------------------
float Wvisc_dr(float rlen, float h, __constant struct SPHParams* params)
// Derivative with respect to |r| divided by |r|
// 
{
    float alpha = 15./(2.*params->PI * h*h*h);
    float rh = rlen / h;
    float Wij = (-1.5*rh + 2.)/(h*h) - 0.5/(rh*rlen*rlen);
    return Wij;
}
//----------------------------------------------------------------------
float Wvisc_lapl(float rlen, float h, __constant struct SPHParams* params)
{
    /*
    float h3 = h*h*h;
    float alpha = 45./(params->PI * h3*h3); 
    float Wij = alpha*(h-rlen);
    return Wij;
    */
    return h - rlen;
}
//----------------------------------------------------------------------

//_WPOLY6_CL_
#endif 
