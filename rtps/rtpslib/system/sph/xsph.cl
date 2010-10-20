#define STRINGIFY(A) #A

//do the SPH pressure calculations and update the force
std::string xsph_program_source = STRINGIFY(

typedef struct SPHParams
{
    float4 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float4 grid_max;            //so we have padding in C++ definition
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
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

       
__kernel void xsph(__global float4* pos, __global float4* veleval, __global float* density, __global float4* force, __global float4* xsph, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);

    float sadf = 5;
    float4 p = pos[i] * params->simulation_scale;
    float4 v = veleval[i];
    float di = density[i];
 
    //obviously this is going to be passed in as a parameter (rather we will use neighbor search)
    int num = get_global_size(0);
    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h9 = h*h*h * h*h*h * h*h*h;
    float alpha = 315.f / 64.0f / params->PI / h9;

    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        if(j == i) continue;
        float4 pj = pos[j] * params->simulation_scale;
        float4 r = p - pj;
        float rlen = magnitude(r);
        if(rlen < h)
        {
            float r2 = rlen*rlen;
            float re2 = h*h;
            if(r2/re2 <= 4.f)
            {
                float4 vj = veleval[j];
                float dj = density[j];

                float hr2 = (h*h - dist_squared(r));
                float Wij = alpha * hr2*hr2*hr2;
                float fc = 2.0 * params->mass * Wij / (di + dj);
                f += fc * (vj - v);
                //f = (float4)(fc, fc, fc, 0.0f);
            }

        }
    }
    xsph[i] = f;

}
);

