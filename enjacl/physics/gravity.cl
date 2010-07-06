//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global float4* generators, __global float4* velocities, __global float* life, float h)

{
    unsigned int i = get_global_id(0);

    h = h*10;
    life[i] -= h;    //should probably depend on time somehow
    if(life[i] <= 0.)
    {
        //reset this particle
        vertices[i].x = generators[i].x;
        vertices[i].y = generators[i].y;
        vertices[i].z = generators[i].z;

        velocities[i].x = 0.f;
        velocities[i].y = 0.f;
        velocities[i].z = 0.f;
        life[i] = 1.;
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

     
    colors[i].x = 1.f;
    colors[i].y = life[i];
    colors[i].z = life[i];
    colors[i].w = 1-life[i];
}


