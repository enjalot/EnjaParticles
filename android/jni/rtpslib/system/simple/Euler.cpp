#include "../Simple.h"
#include <math.h>

namespace rtps {

#if 0
void SPH::loadEuler()
{
    #include "euler.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_euler = Kernel(ps->cli, euler_program_source, "euler");
  
    //TODO: fix the way we are wrapping buffers
    k_euler.setArg(0, cl_position.cl_buffer[0]);
    k_euler.setArg(1, cl_velocity.cl_buffer[0]);
    k_euler.setArg(2, cl_force.cl_buffer[0]);
    k_euler.setArg(3, ps->settings.dt); //time step
    k_euler.setArg(4, cl_params.cl_buffer[0]);

} 
#endif

float distance(float4 p1, float4 p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

float length(float4 v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

//normalized vector pointing from p1 to p2
float4 norm_dir(float4 p1, float4 p2)
{
    float4 dir = float4(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, 0.0f);
    float norm = length(dir);
    if(norm > 0)
    {
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    }
    return dir;
}

float4 force_field(float4 p, float4 ff, float dist, float max_force)
{
    float d = distance(p, ff);
    if(d < 14)
    {
        float4 dir = norm_dir(p, ff);
        float mag = max_force * (dist - d)/dist;
        dir.x *= mag;
        dir.y *= mag;
        return dir;
    }
    return float4(0, 0, 0, 0);
}

void Simple::cpuEuler()
{
    //printf("in cpuEuler\n");
    float h = ps->settings.dt;
    //printf("h: %f\n", h);

    float4 f1 = float4(20, 10, 0, 0);
    float4 f2 = float4(-5, 50, 0, 0);


    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];

        /*
		if (i == 0) {
			printf("==================================\n");
			printf("Euler: p[%d]= %d, %f, %f, %f\n", i, p.x, p.y, p.z, p.w);
			printf("       v[%d]= %f, %f, %f, %f\n", i, v.x, v.y, v.z, v.w);
		}
        */

        //external force is gravity
        //f.z += -9.8f;

        /*
        float speed = magnitude(f);
        if(speed > 600.0f) //velocity limit, need to pass in as struct
        {
            f.x *= 600.0f/speed;
            f.y *= 600.0f/speed;
            f.z *= 600.0f/speed;
        }
        */

        float4 ff1 = force_field(p, f1, 14.0f, 15.0f);
        float4 ff2 = force_field(p, f2, 14.0f, 15.0f);

        f.x += ff1.x + ff2.x;
        f.y += ff1.y + ff2.y;

        v.x += h*f.x;
        v.y += h*f.y;
        v.z += h*f.z;
        
        p.x += h*v.x;
        p.y += h*v.y;
        p.z += h*v.z;
        p.w = 1.0f; //just in case

        velocities[i] = v;
        positions[i] = p;
    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}
