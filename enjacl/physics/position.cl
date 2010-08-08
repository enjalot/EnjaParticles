#define STRINGIFY(A) #A

std::string position_program_source = STRINGIFY(
//update the particle position and color
__kernel void pos_update(__global float4* vertices, __global float4* vert_gen, __global float4* velocities, __global float4* transform, float h)

{
    unsigned int i = get_global_id(0);

    //h = h*10;
    float life = velocities[i].w;
    if(life == 1.f) //particles have been reset by vel_update kernel
    {
        //reset this particle's position
        //with current transform
        float4 pos = vert_gen[i];
        //transform pos to get global coordinates
        //3x3 matrix multiply followed by vector add
        float4 pos_t = (float4)(dot(transform[0], pos), dot(transform[1], pos), dot(transform[2], pos), 0);
        pos = pos_t + transform[3];

        vertices[i] = pos;
    /*
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;
    */
    } 
    vertices[i].x += h*velocities[i].x;
    vertices[i].y += h*velocities[i].y;
    vertices[i].z += h*velocities[i].z;
}
);

