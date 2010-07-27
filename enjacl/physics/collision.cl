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

        velocities[i].x = 1.f;
        velocities[i].y = 0.f;
        velocities[i].z = 1.f;
        life[i] = 1.;
    } 
    float xn = vertices[i].x;
    float yn = vertices[i].y;
    float zn = vertices[i].z;


    float vxn = velocities[i].x;
    float vyn = velocities[i].y;
    float vzn = velocities[i].z;
    velocities[i].x = vxn;
    velocities[i].y = vyn - h*9.8;
    velocities[i].z = vzn;// - h*9.8;

    xn += h*velocities[i].x;
    yn += h*velocities[i].y;
    zn += h*velocities[i].z;

    //plane
    if (yn < -2.0f)
    {
        float4 normal = (-0.707106, 0.707106, 0.0, 0.0); //this is actually a unit vector
        normal = normalize(normal);
        float4 vel = velocities[i];
        float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
        vel /= mag;
        vel = 2.f*(dot(normal, vel))*normal - vel;
        //vel *= mag; //we know the direction lets go the right speed
        xn = vertices[i].x + h*vel.x;
        yn = vertices[i].y + h*vel.y;
        zn = vertices[i].z + h*vel.z;
        velocities[i] = vel;
    }

    vertices[i].x = xn;
    vertices[i].y = yn;
    vertices[i].z = zn;

     
    colors[i].x = 1.f;
    colors[i].y = life[i];
    colors[i].z = life[i];
    //colors[i].w = 1-life[i];
    colors[i].w = 1;
}


/*
float4 normalize(float4 v)
{
    //v is a 4 vector but we only use the x,y,z components
    float magnitude = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v = v/magnitude;
    //for vertices we don't want w component influencing other calculations
    v.w = 1.0f; 
    return v;
}
*/


