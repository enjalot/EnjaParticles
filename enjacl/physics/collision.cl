//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, float h)
{
    unsigned int i = get_global_id(0);

    float life = velocities[i].w;
    life -= h;    //should probably depend on time somehow
    h = h*10;
    if(life <= 0.)
    {
        //reset this particle
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;

        velocities[i].x = velo_gen[i].x;
        velocities[i].y = velo_gen[i].y;
        velocities[i].z = velo_gen[i].z;
        life = 1.;
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
    
    //particle and plane collision
    if (yn < -1.0f)
    {

        float4 posA = vertices[i];
        float radiusA = 5.0f;
        float4 posB = vertices[i] + radiusA/2.0f;
        float4 relPos = (float4)(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0);
        float dist = sqrt(relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
        float collideDist = radiusA + 0.0;//radiusB;


        //float4 norm = (-0.707106, 0.707106, 0.0, 0.0); //this is actually a unit vector
        float4 norm = (float4)(0.0, 1.0, 0.0, 0.0); 
        norm = normalize(norm);
        float4 velA = velocities[i];    //velocity of particle
        float4 velB = (float4)(0,0,0,0);  //velocity of object
        float4 relVel = (float4)(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0);

        float relVelDotNorm = relVel.x * norm.x + relVel.y * norm.y + relVel.z * norm.z;
        float4 tanVel = (float4)(relVel.x - relVelDotNorm * norm.x, relVel.y - relVelDotNorm * norm.y, relVel.z - relVelDotNorm * norm.z, 0);
        float4 force = (float4)(0,0,0,0);
        float springFactor = -.5;//-spring * (collideDist - dist);
        float damping = 1.0f;
        float shear = 1.0f;
        float attraction = 1.0f;
        force = (float4)(
            springFactor * norm.x + damping * relVel.x + shear * tanVel.x + attraction * relPos.x,
            springFactor * norm.y + damping * relVel.y + shear * tanVel.y + attraction * relPos.y,
            springFactor * norm.z + damping * relVel.z + shear * tanVel.z + attraction * relPos.z,
            0
        );

        //float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
        //vel /= mag;
        //vel = 2.f*(dot(normal, vel))*normal - vel;
        ////vel *= mag; //we know the direction lets go the right speed
        
        velA += force;
        
        xn = vertices[i].x + h*velA.x;
        yn = vertices[i].y + h*velA.y;
        zn = vertices[i].z + h*velA.z;
        velocities[i] = velA;
    }

    vertices[i].x = xn;
    vertices[i].y = yn;
    vertices[i].z = zn;

     
    colors[i].x = 1.f;
    colors[i].y = life;
    colors[i].z = life;
    //colors[i].w = 1-life;
    colors[i].w = 1;
    
    //save the life!
    velocities[i].w = life;
}



