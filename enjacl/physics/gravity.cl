//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, float h)

{
    unsigned int i = get_global_id(0);

    h = h*10;
    float life = velocities[i].w;
    life -= h;
    if(life <= 0.)
    {
        //reset this particle
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;
        vertices[i].w = 0.4f;

        velocities[i].x = 0.0f;//velo_gen[i].x;
        velocities[i].y = 0.0f;//velo_gen[i].y;
        velocities[i].z = 0.0f;//velo_gen[i].z;
        life = 1.0f;
    } 
    float xn = vertices[i].x;
    float yn = vertices[i].y;
    float zn = vertices[i].z;


    float vxn = velocities[i].x;
    float vyn = velocities[i].y;
    float vzn = velocities[i].z;
    velocities[i].x = vxn;
    velocities[i].y = vyn;// - h*9.8;
    velocities[i].z = vzn - h*9.8;

    vertices[i].x = xn + h*velocities[i].x; //xn + h*(sigma * (yn - xn));
    vertices[i].y = yn + h*velocities[i].y; //yn + h*(xn*(rho - zn));
    vertices[i].z = zn + h*velocities[i].z; // + h*(xn*yn - beta * zn);

     
    colors[i].x = life - .2f;
    colors[i].y = 0.0f - life * .8f;
    colors[i].z = 0.0f - life;
    colors[i].w = life;
    
    //save the life!
    velocities[i].w = life;
}


