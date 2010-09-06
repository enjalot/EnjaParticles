#define STRINGIFY(A) #A

//do the SPH pressure calculations and update the force
std::string collision_wall_program_source = STRINGIFY(

//from Krog '10
float4 calculateRepulsionForce(float4 normal, float4 vel, float boundary_stiffness, float boundary_dampening, float boundary_distance)
{
    //TODO
    //right now these are the default values for SimpleSPH in Krog's SPHSimLib
    //to be replaced with a parameter struct passed in
    vel.w = 0.0f;
    float4 repulsion_force = (boundary_stiffness * boundary_distance - boundary_dampening * dot(normal, vel))*normal;
    return repulsion_force;
}


__kernel void collision_wall(__global float4* pos, __global float4* vel,  __global float4* force)//, __global float4* walls)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];
    float4 r_f = (float4)(0.f, 0.f, 0.f, 0.f);

    //TODO paramater struct, grid walls passed in
    //we should have a grid data structure passed in with min/max to calculate these things
    float boundary_stiffness = 20000.0f;
    float boundary_dampening = 256.0f;
    float rest_distance = 0.025641;
    float boundary_distance = rest_distance * .5f;
    float EPSILON = .00001f;
    
    //bottom wall
    float diff = boundary_distance - (-100.0f - p.z);
    if (diff < EPSILON)
    {
        float4 normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, boundary_stiffness, boundary_dampening, boundary_distance);
    }
    force[i] += r_f;
    //TODO add more walls

    //TODO add friction forces


}
);

