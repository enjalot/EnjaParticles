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


__kernel void collision_wall(__global float4* pos, __global float4* vel,  __global float4* force, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i] * params->simulation_scale;
    float4 r_f = (float4)(0.f, 0.f, 0.f, 0.f);

    //bottom wall
    float diff = params->boundary_distance - (p.z - params->grid_min.z) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
        //r_f += calculateRepulsionForce(normal, v, boundary_stiffness, boundary_dampening, boundary_distance);
    }

    //Y walls
    diff = params->boundary_distance - (p.y - params->grid_min.y) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }
    diff = params->boundary_distance - (params->grid_max.y - p.y) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(0.0f, -1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }
    //X walls
    diff = params->boundary_distance - (p.x - params->grid_min.x) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }
    diff = params->boundary_distance - (params->grid_max.x - p.x) * params->simulation_scale;
    if (diff > params->EPSILON)
    {
        float4 normal = (float4)(-1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    }


    //TODO add friction forces

    force[i] += r_f;

}
);

