#define STRINGIFY(A) #A

std::string leapfrog_program_source = STRINGIFY(

        
typedef struct SPHParams
{
    float4 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float4 grid_max;            //so we have padding in C++ definition
//    int num;
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound

 
} SPHParams;


 
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       
        
__kernel void leapfrog(__global float4* pos, __global float4* vel, __global float4* veleval, __global float4* force, __global float4* xsph, __global float4* color, float h, __constant struct SPHParams* params )
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    //external force is gravity
    f.z += -9.8f;

    float speed = magnitude(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    float4 vnext = v + h*f;
    vnext += .1f * xsph[i];//should be param XSPH factor
    p += h * vnext / params->simulation_scale;
    p.w = 1.0f; //just in case

    veleval[i] = (v + vnext) * .5f;

    vel[i] = vnext;
    pos[i] = p;

    //float factor = (p.z - params->grid_min.z)/(params->grid_max.z - params->grid_min.z);
    //color[i].x = factor;
    //color[i].z = 1.0f - factor;
    color[i] = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
}
);

