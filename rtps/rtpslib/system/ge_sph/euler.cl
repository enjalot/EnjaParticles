// Put in cl_macros.h
#define DENS 0
#define POS 1
#define VEL 2
#define FOR 3

#define numParticles num

#define FETCH(t, i) t[i]
#define FETCH_VEL(t, i) 	t[i+VEL*numParticles]
#define FETCH_POS(t, i) 	t[i+POS*numParticles]
#define FETCH_DENS(t, i) 	t[i+DENS*numParticles]
#define FETCH_FOR(t, i) 	t[i+FOR*numParticles]
#define pos(i) 		vars_sorted[i+POS*numParticles]
#define vel(i) 		vars_sorted[i+VEL*numParticles]
#define density(i) 	vars_sorted[i+DENS*numParticles].x
#define force(i) 	vars_sorted[i+FOR*numParticles]

//----------------------------------------------------------------------
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       
        
//----------------------------------------------------------------------
__kernel void ge_euler(__constant nb_vars, __global float4* vars_sorted, __constant float h)
{
    unsigned int i = get_global_id(0);
	int num = get_global_size(0);

    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);

    //external force is gravity
    f.z += -9.8f;

    float speed = magnitude(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    v += h*f;
    p += h*v;
    p.w = 1.0f; //just in case

    vel(i) = v;
    pos(i) = p;
}
//----------------------------------------------------------------------
#if 0
__kernel void euler(__global float4* pos, __global float4* vel, __global float4* force, float h)
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

    v += h*f;
    p += h*v;
    p.w = 1.0f; //just in case

    vel[i] = v;
    pos[i] = p;

}
#endif
//----------------------------------------------------------------------
