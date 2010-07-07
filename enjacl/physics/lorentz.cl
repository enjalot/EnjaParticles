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

    float sigma = 10;
    float beta = 8./3.;
    //float rho = 99.96;
    float rho = 28;

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

     
    colors[i].x = 1.f;
    colors[i].y = life[i]*.5;
    colors[i].z = life[i];
    colors[i].w = 1.-life[i];
}


