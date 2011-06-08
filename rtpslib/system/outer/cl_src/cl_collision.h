#ifndef _CL_COLLISION_H_
#define _CL_COLLISION_H_


//from Krog '10
float4 calculateRepulsionForce(float4 normal, float4 vel, float boundary_stiffness, float boundary_dampening, float distance)
{
    vel.w = 0.0f;
    float4 repulsion_force = (boundary_stiffness * distance - boundary_dampening * dot(normal, vel))*normal;
    repulsion_force.w = 0.0f;
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


#endif
