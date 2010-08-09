#define STRINGIFY(A) #A

std::string vfield_program_source = STRINGIFY(
float4 vfield(float4 yn)
{
    float4 vn;
    vn.x = 2.0f*sin(yn.y) + 1.0f * cos(yn.z);
    vn.y = 2.0f*cos(yn.x);
    vn.z = yn.z - 3.0f;
    vn.w = yn.w;
    return vn;
}

//Forward Euler
void forward_euler(float4 yn, __global float4* vn, unsigned int i, float h)
{
    //calculate the velocities from the lorentz attractor equations
    vn[i] = vfield(yn);
}

//RK4
void runge_kutta(float4 yn, __global float4* vn, unsigned int i, float h)
{
    float4 k1 = vfield(yn); 
    float4 k2 = vfield(yn + .5f*h*k1);
    float4 k3 = vfield(yn + .5f*h*k2);
    float4 k4 = vfield(yn + h*k3);

    vn[i] = (k1 + 2.f*k2 + 2.f*k3 + k4)/6.f;
}
//update the particle position and color
__kernel void vel_update(__global float4* vertices, __global float4* colors, __global float4* velo_gen, __global float4* velocities, __global float4* transform, float h)

{
    unsigned int i = get_global_id(0);
    float life = velocities[i].w;
    life -= h/2;    //should probably depend on time somehow
    if(life <= 0.)
    {
        //reset this particle
        float4 vel = velo_gen[i];
        vel = (float4)(dot(transform[0], vel), dot(transform[1], vel), dot(transform[2], vel), 0);
        //vel = vel_t + transform[3];

        velocities[i] = 5*vel;
        /*
        velocities[i].x = 5*velo_gen[i].x;
        velocities[i].y = 5*velo_gen[i].y;
        velocities[i].z = 5*velo_gen[i].z;
        */
 
        life = 1.0f;
    } 

    float4 pos = vertices[i];
    //forward_euler(vertices, velocities, i, h); 
    runge_kutta(pos, velocities, i, h*1.f); //runge_kutta can handle a bigger time-step

/*     
    colors[i].x = 1.f;
    colors[i].y = life;
    colors[i].z = life;
    colors[i].w = 1.-life;
*/
    //colors[i].x = life - .2f;
    colors[i].x = 1.0f;
    colors[i].y = 1.0f - 10.0f*life;// - life;
    colors[i].z = 1.0f - 10.0f*life;/// - life;
    colors[i].w = life;
 
    
    //save the life!
    velocities[i].w = life;
}
);
