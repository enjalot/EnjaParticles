#ifndef _WPOLY6_CL_
#define _WPOLY6_CL_

//----------------------------------------------------------------------
float Wpoly6(float4 r, float h, __constant struct SPHParams* params)
{
    float r2 = r.x*r.x + r.y*r.y + r.z*r.z;  // dist_squared(r);
	float h9 = h*h;
	float hr2 = (h9-r2); // h9 = h^2
	h9 = h9*h;   //  h9 = h^3
    float alpha = 315.f/64.0f/params->PI/(h9*h9*h9);
    float Wij = alpha * hr2*hr2*hr2;
    return Wij;
}
//----------------------------------------------------------------------
float Wspiky(float rlen, float h, __constant struct SPHParams* params)
{
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;
	float hr2 = (h - rlen);
	float Wij = alpha * hr2*hr2*hr2/rlen;
	return Wij;
}
//----------------------------------------------------------------------

#endif _WPOLY6_CL_
