#define STRINGIFY(A) #A

std::string lorenz_program_source = STRINGIFY(
//Lorenz Equations
__constant float sigma = 10.f;
__constant float beta = 8.f/3.f;
//__constant float rho = 28.f;
__constant float rho = 99.96f;

float lorenzX(float xn, float yn)
{
    return sigma * (yn - xn);
}
float lorenzY(float xn, float yn, float zn)
{
    return xn*(rho - zn) - yn;
}
float lorenzZ(float xn, float yn, float zn)
{
    return (xn*yn - beta * zn);
}

float4 lorenz(float4 yn)
{
    float4 vn;
    vn.x = lorenzX(yn.x, yn.y);
    vn.y = lorenzY(yn.x, yn.y, yn.z);
    vn.z = lorenzZ(yn.x, yn.y, yn.z);
    vn.w = yn.w;
    return vn;
}

//Forward Euler
void forward_euler(float4 yn, __global float4* vn, unsigned int i, float h)
{
    //calculate the velocities from the lorenz attractor equations
    vn[i] = lorenz(yn);

    /*
    //update the positions with the new velocities
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
    */
}

//RK4
void runge_kutta(float4 yn, __global float4* vn, unsigned int i, float h)
{
    float4 k1 = lorenz(yn); 
    float4 k2 = lorenz(yn + .5f*h*k1);
    float4 k3 = lorenz(yn + .5f*h*k2);
    float4 k4 = lorenz(yn + h*k3);

    vn[i] = (k1 + 2.f*k2 + 2.f*k3 + k4)/6.f;
    
    /*
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
    */
}

//update the particle position and color
__kernel void vel_update(__global float4* vertices, __global float4* colors, __global float4* velo_gen, __global float4* velocities, float h)
{
    unsigned int i = get_global_id(0);
    float life = velocities[i].w;
    life -= h/10.;    //should probably depend on time somehow
    if(life <= 0.)
    {
        //reset this particle

        velocities[i].x = velo_gen[i].x;
        velocities[i].y = velo_gen[i].y;
        velocities[i].z = velo_gen[i].z;
        life = 1.0f;
    } 

    //forward_euler(vertices, velocities, i, h); 
    float4 pos = vertices[i];
    runge_kutta(pos, velocities, i, h*1.f); //runge_kutta can handle a bigger time-step

     
    colors[i].x = 1.f;
    colors[i].y = life*.5;
    colors[i].z = life;
    colors[i].w = 1.-life;
    
    //save the life!
    velocities[i].w = life;
}
);

