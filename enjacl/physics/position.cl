#define STRINGIFY(A) #A

std::string position_program_source = STRINGIFY(
//update the particle position and color
__kernel void pos_update(__global float4* vertices, __global float4* vert_gen, __global float4* velocities, float h)

{
    unsigned int i = get_global_id(0);

    //h = h*10;
    float life = velocities[i].w;
    if(life == 1.f) //particles have been reset by vel_update kernel
    {
        //reset this particle's position
        //with current transform
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;
    } 
    vertices[i].x += h*velocities[i].x;
    vertices[i].y += h*velocities[i].y;
    vertices[i].z += h*velocities[i].z;
}
);

