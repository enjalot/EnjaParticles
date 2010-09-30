#define STRINGIFY(A) #A

//do the SPH pressure calculations and update the force
std::string pressure_program_source = STRINGIFY(

typedef struct SPHParams
{
    float3 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float3 grid_max;            //so we have padding in C++ definition
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

       
__kernel void pressure(__global float4* pos, __global float* density, __global float4* force, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);
    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 p = pos[i] * params->simulation_scale;

    //obviously this is going to be passed in as a parameter
    int num = 1024;
    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;


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
                //float R = sqrt(r2/re2);
                //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                float hr2 = (h - rlen);
                float Wij = alpha * hr2*hr2*hr2/rlen;
                //from tim's code
                /*
                float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                float kern = params->mass * Wij * (Pi + Pj) / (density[i] * density[j]);
                */
                //form simple SPH in Krog's thesis
                float Pi = params->K*(density[i] - 1000.0f); //rest density
                float Pj = params->K*(density[j] - 1000.0f); //rest density
                float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (2.0f * density[j]);
                //float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (density[i] * density[j]);
                f += kern * r;
            }

        }
    }
    force[i] = f;

}
);

