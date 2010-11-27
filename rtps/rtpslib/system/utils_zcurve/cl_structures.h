// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_STRUCTURES_H_
#define _CL_STRUCTURES_H_

//----------------------------------------------------------------------
struct GPUReturnValues
{
	int compact_size;
};
//----------------------------------------------------------------------
struct CellOffsets
{
	int4 offsets[32];
};
//----------------------------------------------------------------------
// Will be local variable
// used to output multiple variables per point
typedef struct PointData
{
	// density.x: density
	// density.y: denominator: sum_i (m_j/rho_j W_j)
	float4 density;
	float4 color;  // x component
	float4 color_normal;
	float4 color_lapl;
	float4 force;
	float4 surf_tens;
	float4 xsph;
} PointData;
//----------------------------------------------------------------------
struct GridParamsScaled
// scaled with simulation_scale
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    float4          bnd_min;
    float4          bnd_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    float4          grid_inv_delta;
	int4			expo; // grid_res = 2^expo
	int4			shift[27]; // neighbors
    int             numParticles; // nb fluid particles: wrong spot for this variable
    int             nb_vars; // for combined variables (vars_sorted, etc.)
    int             nb_points; // total number of grid points
};
//----------------------------------------------------------------------
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    float4          bnd_min;
    float4          bnd_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    float4          grid_inv_delta;
	int4			expo; // grid_res = 2^expo
	int4			shift[27]; // neighbors
    int             numParticles; // nb fluid particles: wrong spot for this variable
    int             nb_vars; // for combined variables (vars_sorted, etc.)
    int             nb_points; // total number of grid points
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
    float rest_density;
    float smoothing_distance;
    float particle_radius;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound
	float dt;

#if 1
	float wpoly6_coef;
	float wpoly6_d_coef;
	float wpoly6_dd_coef; // laplacian
	float wspike_coef;
	float wspike_d_coef;
	float wspike_dd_coef;
	float wvisc_coef;
	float wvisc_d_coef;
	float wvisc_dd_coef;
#endif
};
//----------------------------------------------------------------------


#endif
