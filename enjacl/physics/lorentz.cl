//Lorentz Equations
float sigma = 10.f;
float beta = 8.f/3.f;
float rho = 28.f;
float lorentzX(float xn, float yn)
{
    return sigma * (yn - xn);
}
float lorentzY(float xn, float zn)
{
    return xn*(rho - zn);
}
float lorentzZ(float xn, float yn, float zn)
{
    return (xn*yn - beta * zn);
}

//Forward Euler
void forward_euler(__global float4* yn, __global float4* vn, unsigned int i, float h)
{
    float4 ynt;
    //calculate the velocities from the lorentz attractor equations
    vn[i].x = lorentzX(yn[i].x, yn[i].y);
    vn[i].y = lorentzY(yn[i].x, yn[i].z);
    vn[i].z = lorentzZ(yn[i].x, yn[i].y, yn[i].z);

    //update the positions with the new velocities
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
}

//RK4


//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global float4* generators, __global float4* velocities, __global float* life, float h)

{
    unsigned int i = get_global_id(0);

    life[i] -= h/10.;    //should probably depend on time somehow
    if(life[i] <= 0.)
    {
        //reset this particle
        vertices[i].x = generators[i].x;
        vertices[i].y = generators[i].y;
        vertices[i].z = generators[i].z;

        velocities[i].x = 0.0f;
        velocities[i].y = 0.0f;
        velocities[i].z = 0.0f;
        life[i] = 1.;
    } 

    forward_euler(vertices, velocities, i, h); 

     
    colors[i].x = 1.f;
    colors[i].y = life[i]*.5;
    colors[i].z = life[i];
    colors[i].w = 1.-life[i];
}

//This code used to be in the kernel
/*
    float xn = vertices[i].x;
    float yn = vertices[i].y;
    float zn = vertices[i].z;

    //h = .001;
    float vxn = velocities[i].x;
    float vyn = velocities[i].y;
    float vzn = velocities[i].z;
    velocities[i].x = sigma * (yn - xn);
    velocities[i].y = xn*(rho - zn);
    velocities[i].z = (xn*yn - beta * zn);// + vzn - h*9.8;

    vertices[i].x = xn + h*velocities[i].x; //xn + h*(sigma * (yn - xn));
    vertices[i].y = yn + h*velocities[i].y; //yn + h*(xn*(rho - zn));
    vertices[i].z = zn + h*velocities[i].z; // + h*(xn*yn - beta * zn);
*/

