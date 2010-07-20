
float4 vfield(float4 yn)
{
    float4 vn;
    vn.x = 2.0f*sin(yn.y);// + 1.0f * cos(yn.z);
    vn.y = 2.0f*cos(yn.x);// + 1.0f * cos(yn.z);
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
__kernel void enja(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, float h)

{
    unsigned int i = get_global_id(0);
    float life = velocities[i].w;
    life -= h;
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
    /*
    float vxn = velocities[i].x;
    float vyn = velocities[i].y;
    float vzn = velocities[i].z;
    velocities[i].x = vxn + h*(cos(vyn));
    //velocities[i].y = vyn;// - h*9.8;
    velocities[i].y = vyn + h*(sin(vxn));
    //velocities[i].y = vyn + h*9.8*(colors[i].x + colors[i].y + colors[i].z)/3.0f;
    //velocities[i].z = vzn - h*9.8; //exagerate the effect of gravity for now

    vertices[i].x = xn + h*velocities[i].x; //xn + h*(sigma * (yn - xn));
    vertices[i].y = yn + h*velocities[i].y; //yn + h*(xn*(rho - zn));
    vertices[i].z = zn + h*velocities[i].z; // + h*(xn*yn - beta * zn);
    */

    runge_kutta(vertices, velocities, i, h*1.f); //runge_kutta can handle a bigger time-step
/*     
    colors[i].x = life - .2f;
    colors[i].y = 0.0f - life * .8f;
    colors[i].z = 0.0f - life;
    colors[i].w = life;
*/  
    //save the life!
    velocities[i].w = life;
}


