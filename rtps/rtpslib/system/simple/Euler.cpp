#include "../Simple.h"
#include <math.h>

namespace rtps {

void Simple::loadEuler()
{
    std::string path(SIMPLE_CL_SOURCE_DIR);
    path += "/euler_cl.cl";
    k_euler = Kernel(ps->cli, path, "euler");
  
    k_euler.setArg(0, cl_position.getDevicePtr());
    k_euler.setArg(1, cl_velocity.getDevicePtr());
    k_euler.setArg(2, cl_force.getDevicePtr());
    k_euler.setArg(3, ps->settings.dt); //time step

} 

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

float4 predator_prey(float4 p)
{
    float4 v = float4(0,0,0,0);
    int a1 = 2;
    int a2 = 2;
    int b1 = 1;
    int b2 = 1;
    v.x = a1*p.x - b1*p.x*p.y;
    v.y = -a2*p.y + b2*p.y*p.x;
    //v.x = a1 - b1*p.y;
    //v.y = -a2 + b2*p.x;
    return v;
}

float4 runge_kutta(float4 yn, float h)
{
    float4 k1 = predator_prey(yn); 
    float4 k2 = predator_prey(yn + .5f*h*k1);
    float4 k3 = predator_prey(yn + .5f*h*k2);
    float4 k4 = predator_prey(yn + h*k3);

    float4 vn = (k1 + 2.f*k2 + 2.f*k3 + k4);
    return (1./6.)*vn;
    
    /*
    yn[i].x += h*(vn[i].x);
    yn[i].y += h*(vn[i].y);
    yn[i].z += h*(vn[i].z);
    //yn[i] += h*vn[i]; //this would work with float3
    */
}



float4 force_field(float4 p, float4 ff, float dist, float max_force)
{
    float d = distance(p, ff);
    if(d < dist)
    {
        float4 dir = norm_dir(p, ff);
        float mag = max_force * (dist - d)/dist;
        dir.x *= mag;
        dir.y *= mag;
        dir.z *= mag;
        return dir;
    }
    return float4(0, 0, 0, 0);
}

void Simple::cpuEuler()
{
    //printf("in cpuEuler\n");
    float h = ps->settings.dt;
    //printf("h: %f\n", h);

    float4 f1 = float4(.5, .5, .0, 0);
    float4 f2 = float4(1, 1, .0, 0);
    float4 f3 = float4(1.5, .5, .0, 0);


    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];


        //float4 pp = predator_prey(p);
        float4 pp = runge_kutta(p, h);
        //v = pp;
        float4 ff1 = force_field(p, f1, .84f, 15.0f);
        float4 ff2 = force_field(p, f2, 1.4f, 16.0f);
        float4 ff3 = force_field(p, f3, .8f, 15.0f);

        f += ff1 + ff2 + ff3;

        v.x += h*f.x;
        v.y += h*f.y;
        v.z += h*f.z;
        

        p.x += h*v.x;
        p.y += h*v.y;
        p.z += h*v.z;
        p.w = 1.0f; //just in case

        velocities[i] = v;
        positions[i] = p;

        float colx = v.x;
        float coly = v.y;
        float colz = v.z;
        if(colx < 0) {colx = -1.0f*colx;}
        if(colx > 1) {colx = 1.0f;}
        if(coly < 0) {coly = -1.0f*coly;}
        if(coly > 1) {coly = 1.0f;}
        if(colz < 0) {colz = -1.0f*colz;}
        if(colz > 1) {colz = 1.0f;}

        colors[i].x = colx;
        colors[i].y = coly;
        colors[i].z = colz;

    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}
