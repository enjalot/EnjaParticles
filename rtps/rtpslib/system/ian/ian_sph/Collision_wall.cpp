#include "../SPH.h"

namespace rtps {

void SPH::loadCollision_wall()
{
    #include "collision_wall.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_collision_wall = Kernel(ps->cli, collision_wall_program_source, "collision_wall");
  
    //TODO: fix the way we are wrapping buffers
    k_collision_wall.setArg(0, cl_position.cl_buffer[0]);
    k_collision_wall.setArg(1, cl_velocity.cl_buffer[0]);
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


void SPH::cpuCollision_wall()
{

    for(int i = 0; i < num; i++)
    {
        float scale = params.simulation_scale;
        float4 p = positions[i];
        float4 v = velocities[i];
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        float4 r_f = float4(0.f, 0.f, 0.f, 0.f);
        float4 crf = float4(0.f, 0.f, 0.f, 0.f);

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

        }
        diff = params.boundary_distance - (params.grid_max.y - p.y) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(0.0f, -1.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;

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

        }
        diff = params.boundary_distance - (params.grid_max.x - p.x) * params.simulation_scale;
        if (diff > params.EPSILON)
        {
            float4 normal = float4(-1.0f, 0.0f, 0.0f, 0.0f);
            crf = calculateRepulsionForce(normal, v, params.boundary_stiffness, params.boundary_dampening, params.boundary_distance);
            r_f.x += crf.x;
            r_f.y += crf.y;
            r_f.z += crf.z;

        }


        //TODO add friction forces

        forces[i].x += r_f.x;
        forces[i].y += r_f.y;
        forces[i].z += r_f.z;
    }

}



}
