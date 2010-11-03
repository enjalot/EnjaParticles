#ifndef _CL_STRUCTURES_H_
#define _CL_STRUCTURES_H_


typedef struct SPHParams
{
    float4 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float4 grid_max;            //so we have padding in C++ definition
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound
	
    float friction_coef;
	float restitution_coef;
	float shear;
	float attraction;
	float spring;
	float gravity; // -9.8 m/sec^2

    float wpoly6_coef;
	float wpoly6_d_coef;
	float wpoly6_dd_coef; // laplacian
	float wspiky_coef;
	float wspiky_d_coef;
	float wspiky_dd_coef;
	float wvisc_coef;
	float wvisc_d_coef;
	float wvisc_dd_coef;

    int num;
    int nb_vars; // for combined variables (vars_sorted, etc.)
	int choice; // which kind of calculation to invoke
 
} SPHParams;


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

#endif
