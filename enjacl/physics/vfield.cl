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
void forward_euler(__global float4* yn, __global float4* vn, unsigned int i, float h)
{
    //calculate the velocities from the lorentz attractor equations
    vn[i] = vfield(yn[i]);

    //update the positions with the new velocities
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
}

//RK4
void runge_kutta(__global float4* yn, __global float4* vn, unsigned int i, float h)
{
    float4 k1 = vfield(yn[i]); 
    float4 k2 = vfield(yn[i] + .5f*h*k1);
    float4 k3 = vfield(yn[i] + .5f*h*k2);
    float4 k4 = vfield(yn[i] + h*k3);

    vn[i] = (k1 + 2.f*k2 + 2.f*k3 + k4)/6.f;
    
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
}


//update the particle position and color
//__kernel void enja(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, __global float* life, float h)
__kernel void update(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, float h)

{
    unsigned int i = get_global_id(0);
    float life = velocities[i].w;
    life -= h/2;    //should probably depend on time somehow
    if(life <= 0.)
    {
        //reset this particle
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;

        velocities[i].x = velo_gen[i].x;
        velocities[i].y = velo_gen[i].y;
        velocities[i].z = velo_gen[i].z;
        life = 1.0f;
    } 

    //forward_euler(vertices, velocities, i, h); 
    runge_kutta(vertices, velocities, i, h*1.f); //runge_kutta can handle a bigger time-step

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
