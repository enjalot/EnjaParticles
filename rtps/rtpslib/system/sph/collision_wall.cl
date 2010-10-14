#define STRINGIFY(A) #A

//do the SPH pressure calculations and update the force
std::string collision_wall_program_source = STRINGIFY(

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
 
} SPHParams;

//from Krog '10
float4 calculateRepulsionForce(float4 normal, float4 vel, float boundary_stiffness, float boundary_dampening, float distance)
{
    vel.w = 0.0f;
    float4 repulsion_force = (boundary_stiffness * distance - boundary_dampening * dot(normal, vel))*normal;
    return repulsion_force;
}

//from Krog '10
float4 calculateFrictionForce(float4 vel, float4 force, float4 normal, float friction_kinetic, float friction_static_limit)
{
	float4 friction_force = (float4)(0.0f,0.0f,0.0f,0.0f);
    force.w = 0.0f;
    vel.w = 0.0f;

	// the normal part of the force vector (ie, the part that is going "towards" the boundary
	float4 f_n = force * dot(normal, force);
	// tangent on the terrain along the force direction (unit vector of tangential force)
	float4 f_t = force - f_n;

	// the normal part of the velocity vector (ie, the part that is going "towards" the boundary
	float4 v_n = vel * dot(normal, vel);
	// tangent on the terrain along the velocity direction (unit vector of tangential velocity)
	float4 v_t = vel - v_n;

	if((v_t.x + v_t.y + v_t.z)/3.0f > friction_static_limit)
		friction_force = -v_t;
	else
		friction_force = friction_kinetic * -v_t;

	// above static friction limit?
//  	friction_force.x = f_t.x > friction_static_limit ? friction_kinetic * -v_t.x : -v_t.x;
//  	friction_force.y = f_t.y > friction_static_limit ? friction_kinetic * -v_t.y : -v_t.y;
//  	friction_force.z = f_t.z > friction_static_limit ? friction_kinetic * -v_t.z : -v_t.z;

	//TODO; friction should cause energy/heat in contact particles!
	//friction_force = friction_kinetic * -v_t;

	return friction_force;

}


__kernel void collision_wall(__global float4* pos, __global float4* vel,  __global float4* force, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];// * params->simulation_scale;
    float4 f = force[i];
    float4 r_f = (float4)(0.f, 0.f, 0.f, 0.f);
    float4 f_f = (float4)(0.f, 0.f, 0.f, 0.f);

    //these should be moved to the params struct
    //but set to 0 in both of Krog's simulations...
    float friction_kinetic = 0.0f;
    float friction_static_limit = 0.0f;

    //bottom wall
    float diff = params->boundary_distance - (p.z - params->grid_min.z) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
        //r_f += calculateRepulsionForce(normal, v, boundary_stiffness, boundary_dampening, boundary_distance);
    }

    //Y walls
    diff = params->boundary_distance - (p.y - params->grid_min.y) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }
    diff = params->boundary_distance - (params->grid_max.y - p.y) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, -1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }
    //X walls
    diff = params->boundary_distance - (p.x - params->grid_min.x) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }
    diff = params->boundary_distance - (params->grid_max.x - p.x) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(-1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
    }


    force[i] += r_f + f_f;

}
);

