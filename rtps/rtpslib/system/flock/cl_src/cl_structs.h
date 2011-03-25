#ifndef _CL_STRUCTURES_H_
#define _CL_STRUCTURES_H_

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
	float4 xflock;
//	float4 center_of_mass;
//	int num_neighbors;
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

    int nb_cells;
};

struct FLOCKParameters
{

    float4 grid_min;
    float4 grid_max;
    
    float rest_distance;
    float smoothing_distance;
    
    int num;
    int nb_vars; // for combined variables (vars_sorted, etc.)
	int choice; // which kind of calculation to invoke
    
    // Boids
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    
    float w_sep;
    float w_align;
    float w_coh;
};

#endif
