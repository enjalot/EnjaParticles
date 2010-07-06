//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global float4* generators, __global float4* velocities, __global float* life, float dt)

/*
//__kernel void enja(__global float4* vertices, float dt)
//__kernel void enja(__global float4* vertices, __global float4* colors, __global float4* generators, float dt)
//__kernel void enja(__global float4* vertices, __global float4* colors, float dt)
*/

{
    unsigned int i = get_global_id(0);
	return;


    life[i] -= .05;    //should probably depend on time somehow
	#if 1
    if(life[i] <= 0.)
    {
        //reset this particle
        vertices[i].x = generators[i].x;
        vertices[i].y = generators[i].y;
        vertices[i].z = generators[i].z;
        float notrandom = .001f * dt / 5000.0f;//hardcoded based on runlength right now
        velocities[i].z = .001 + notrandom; //not random but oh well
        life[i] = 1.;
    }  
	#endif

    vertices[i].x += velocities[i].x;
    vertices[i].y += velocities[i].y;
    vertices[i].z += velocities[i].z;
    //vertices[i].w += velocities[i].w;
    //velocities[i].z -= .0007; //this needs to depend on time or life


	#if 0
    colors[i].x = 1.0;
    colors[i].y = life[i];
    colors[i].z = life[i];
    colors[i].w = 1.-life[i];
	#endif

/*
    vertices[i].x += dt;
    colors[i].y += dt*10;
    colors[i].z += dt*10;
    colors[i].w -= dt*10;
*/


/*
// old testing stuff
    vertices[i].x = tv.x + 0.01f;
    vertices[i].y = tv.y + 0.01f;
    vertices[i].z = tv.z + 0.01f;    
    //vertices[i].w = 1.f;

    colors[i].y = tc.y - 0.05f;
    colors[i].z = tc.z - 0.05f;
    if(colors[i].y <= 0.0f)
    {
        colors[i].y = 1.f;
    }
    if(colors[i].z <= 0.0f)
    {
        colors[i].z = 1.f;
    }
*/

}


