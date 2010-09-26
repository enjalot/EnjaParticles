// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_STRUCTURES_H_
#define _CL_STRUCTURES_H_

//----------------------------------------------------------------------
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    int            numParticles; // nb fluid particles: wrong spot for this variable
};
//----------------------------------------------------------------------
struct FluidParams
{
	float smoothing_length; // SPH radius
	float scale_to_simulation;
	float mass;
	float dt; // Time step, not necessarily best location
	float friction_coef;
	float restitution_coef;
	float damping;
	float shear;
	float attraction;
	float spring;
	float gravity; // -9.8 m/sec^2
};
//----------------------------------------------------------------------


#endif
