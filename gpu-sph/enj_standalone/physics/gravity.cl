#define STRINGIFY(A) #A

std::string gravity_program_source = STRINGIFY(
//update the particle position and color
__kernel void vel_update(__global float4* vertices, __global float4* colors, __global float4* velo_gen, __global float4* velocities, __global float4* transform, float h)
{
    unsigned int i = get_global_id(0);

   
    //h = h*10;
    float life = velocities[i].w;
    life -= h/5;
    if(life <= 0.)
    {

        float4 vel = velo_gen[i];
        vel = (float4)(dot(transform[0], vel), dot(transform[1], vel), dot(transform[2], vel), 0.0f);
        //vel = vel_t + transform[3];

        velocities[i] = 5.0f*vel;
        //reset this particle
        /*
        velocities[i].x = 5*velo_gen[i].x;
        velocities[i].y = 5*velo_gen[i].y;
        velocities[i].z = 5*velo_gen[i].z;
        */
        life = 1.0f;
    } 
    float vxn = velocities[i].x;
    float vyn = velocities[i].y;
    float vzn = velocities[i].z;
    velocities[i].x = vxn;
    velocities[i].y = vyn;// - h*9.8;
    velocities[i].z = vzn - h*9.8f;
     
    colors[i].x = life;
    colors[i].y = 1.0f - life;
    colors[i].z = 1.0f - life;
    colors[i].w = life;
    
    //save the life!
    velocities[i].w = life;
}
);

