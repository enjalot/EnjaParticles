#include "../SPH.h"

namespace rtps {

void SPH::loadCollision_wall()
{
    #include "collision_wall.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_collision_wall = Kernel(ps->cli, collision_wall_program_source, "collision_wall");
  
    //TODO: fix the way we are wrapping buffers
    k_collision_wall.setArg(0, cl_position.cl_buffer[0]);
    //k_collision_wall.setArg(1, cl_velocity.cl_buffer[0]);
    //should check for leapfrog, then want to use veleval
     if(sph_settings.integrator == LEAPFROG)
    {
        k_collision_wall.setArg(1, cl_veleval.cl_buffer[0]);
    }
    else if(sph_settings.integrator == EULER)
    {
        k_collision_wall.setArg(1, cl_velocity.cl_buffer[0]);
    }
    k_collision_wall.setArg(2, cl_force.cl_buffer[0]);
    k_collision_wall.setArg(3, cl_params.cl_buffer[0]);

} 

float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//from Krog '10
float4 calculateRepulsionForce(float4 normal, float4 vel, float boundary_stiffness, float boundary_dampening, float boundary_distance)
{
    vel.w = 0.0f;
    float coeff = boundary_stiffness * boundary_distance - boundary_dampening * dot(normal, vel);
    float4 repulsion_force = float4(coeff * normal.x, coeff * normal.y, coeff*normal.z, 0.0f);
    return repulsion_force;
}


//from Krog '10
float4 calculateFrictionForce(float4 vel, float4 force, float4 normal, float friction_kinetic, float friction_static_limit)
{
	float4 friction_force = float4(0.0f,0.0f,0.0f,0.0f);

	// the normal part of the force vector (ie, the part that is going "towards" the boundary
    float4 f_n = force;
	f_n.x *= dot(normal, force);
	f_n.y *= dot(normal, force);
	f_n.z *= dot(normal, force);
	// tangent on the terrain along the force direction (unit vector of tangential force)
	float4 f_t = force;
    f_t.x -= f_n.x;
    f_t.y -= f_n.y;
    f_t.z -= f_n.z;

	// the normal part of the velocity vector (ie, the part that is going "towards" the boundary
	float4 v_n = vel;
    v_n.x *= dot(normal, vel);
    v_n.y *= dot(normal, vel);
    v_n.z *= dot(normal, vel);
	// tangent on the terrain along the velocity direction (unit vector of tangential velocity)
	float4 v_t = vel;
    v_t.x -= v_n.x;
    v_t.y -= v_n.y;
    v_t.z -= v_n.z;

	if((v_t.x + v_t.y + v_t.z)/3.0f > friction_static_limit)
    {
		friction_force.x = -v_t.x;
		friction_force.y = -v_t.y;
		friction_force.z = -v_t.z;
    }
	else
    {
		friction_force.x = friction_kinetic * -v_t.x;
		friction_force.y = friction_kinetic * -v_t.y;
		friction_force.z = friction_kinetic * -v_t.z;
    }

	// above static friction limit?
//  	friction_force.x = f_t.x > friction_static_limit ? friction_kinetic * -v_t.x : -v_t.x;
//  	friction_force.y = f_t.y > friction_static_limit ? friction_kinetic * -v_t.y : -v_t.y;
//  	friction_force.z = f_t.z > friction_static_limit ? friction_kinetic * -v_t.z : -v_t.z;

	//TODO; friction should cause energy/heat in contact particles!
	//friction_force = friction_kinetic * -v_t;

	return friction_force;

}



void SPH::cpuCollision_wall()
{

    float4* vel;
    if(sph_settings.integrator == EULER)
    {
        vel = &velocities[0];
    }
    else if(sph_settings.integrator == LEAPFROG)
    {
        vel = &veleval[0];
    }
    for(int i = 0; i < num; i++)
    {
        
        float scale = params.simulation_scale;
        float4 p = positions[i];
        float4 v = vel[i];
        float4 f = forces[i];
        /*
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        */
        float4 r_f = float4(0.f, 0.f, 0.f, 0.f);
        float4 f_f = float4(0.f, 0.f, 0.f, 0.f);
        float4 crf = float4(0.f, 0.f, 0.f, 0.f);
        float4 cff = float4(0.f, 0.f, 0.f, 0.f);

        float friction_kinetic = 0.0f;
        float friction_static_limit = 0.0f;

        //bottom wall
        float diff = params.boundary_distance - (p.z - params.grid_min.z) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            //printf("colliding with the bottom! %d\n", i);
            float4 normal = float4(0.0f, 0.0f, 1.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;
            cff = calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
            f_f.x += cff.x;
            f_f.y += cff.y;
            f_f.z += cff.z;

            //printf("crf %f %f %f \n", crf.x, crf.y, crf.z);
        }

        //Y walls
        diff = params.boundary_distance - (p.y - params.grid_min.y) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(0.0f, 1.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;
            cff = calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
            f_f.x += cff.x;
            f_f.y += cff.y;
            f_f.z += cff.z;

        }
        diff = params.boundary_distance - (params.grid_max.y - p.y) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(0.0f, -1.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;
            cff = calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
            f_f.x += cff.x;
            f_f.y += cff.y;
            f_f.z += cff.z;

        }
        //X walls
        diff = params.boundary_distance - (p.x - params.grid_min.x) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(1.0f, 0.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;
            cff = calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
            f_f.x += cff.x;
            f_f.y += cff.y;
            f_f.z += cff.z;

        }
        diff = params.boundary_distance - (params.grid_max.x - p.x) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(-1.0f, 0.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;
            cff = calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
            f_f.x += cff.x;
            f_f.y += cff.y;
            f_f.z += cff.z;

        }


        //TODO add friction forces

        forces[i].x += r_f.x + f_f.x;
        forces[i].y += r_f.y + f_f.y;
        forces[i].z += r_f.z + f_f.z;
    }

}



}
