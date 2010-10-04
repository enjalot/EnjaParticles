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
    float4          grid_inv_delta;
    int             numParticles; // nb fluid particles: wrong spot for this variable
};
//----------------------------------------------------------------------
struct FluidParams
{
	float smoothing_length; // SPH radius
	float scale_to_simulation;
	//float mass;
	//float dt; // Time step, not necessarily best location
	float friction_coef;
	float restitution_coef;
	float damping;
	float shear;
	float attraction;
	float spring;
	float gravity; // -9.8 m/sec^2
	int choice; // EASY WAY TO SELECT KERNELS
};
//----------------------------------------------------------------------
//pass parameters to OpenCL routines
struct SPHParams
{
    float4 grid_min; // changed by GE
    float4 grid_max;
    float grid_min_padding;     //float3s take up a float4 of space in OpenCL 1.0 and 1.1
    float grid_max_padding;
    float mass;
    float rest_distance;
    float smoothing_distance;
    float particle_radius;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound
};
//----------------------------------------------------------------------


#endif
