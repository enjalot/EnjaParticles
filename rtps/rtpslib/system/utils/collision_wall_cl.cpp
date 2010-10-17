
//do the GE_SPH pressure calculations and update the force

#include "cl_macros.h"
#include "cl_structures.h"


//----------------------------------------------------------------------
//from Krog '10
float4 calculateRepulsionForce(
      float4 normal, 
	  float4 vel, 
	  float boundary_stiffness, 
	  float boundary_dampening, 
	  float boundary_distance)
{
// I am convinced something is wrong either with the pressure, or the Boundary 
// Conditions. Or else something is wrong with the initialization. Not clear about the 
// cell size in relation to fluid. If there are 8 particles per cell, the cell mass is 
// slightly less than the fluid mass (if the cell size = 2*particle_radius). 

    vel.w = 0.0f;  // Removed influence of 4th component of velocity (does not exist)
    float4 repulsion_force = (boundary_stiffness * boundary_distance - boundary_dampening * dot(normal, vel))*normal;
	repulsion_force.w = 0.f;
    return repulsion_force;
}


//----------------------------------------------------------------------
__kernel void collision_wall(
		__global float4* vars_sorted, 
		__constant struct GridParams* gp,
		__constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);
	int num = get_global_size(0);
	int nb_vars = gp->nb_vars;

    float4 p = pos(i); //  pos[i];
    float4 v = veleval(i); //  vel[i];
    float4 r_f = (float4)(0.f, 0.f, 0.f, 0.f);

    //bottom wall
    float diff = params->boundary_distance - (p.z - gp->bnd_min.z);
    if (diff > params->EPSILON)
    {
		// normal points into the domain
        float4 normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }

    //Y walls
    diff = params->boundary_distance - (p.y - gp->bnd_min.y);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }
    diff = params->boundary_distance - (gp->bnd_max.y - p.y);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, -1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }

    //X walls
    diff = params->boundary_distance - (p.x - gp->bnd_min.x);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }
    diff = params->boundary_distance - (gp->bnd_max.x - p.x);
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(-1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }

    //TODO add friction forces

	force(i) += r_f;   //sorted force
}
