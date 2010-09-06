#define STRINGIFY(A) #A

//do the SPH pressure calculations and update the force
std::string pressure_program_source = STRINGIFY(

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
        
__kernel void pressure(__global float4* pos, __global float* density, __global float4* force)
{
    unsigned int i = get_global_id(0);
    force[i] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 p = pos[i];

    //obviously this is going to be passed in as a parameter
    int num = 1024;
    float rest_distance = 0.025641f;
    float rest_density = 1000.0f;
    float smoothing_length = 2.0f * rest_distance;
    float K = 331.3f; //speed of sound

    //stuff from Tim's code (need to match #s to papers)
    float pi = 4.f * atan(1.0f);
    float alpha = 315.f/208.f/pi/smoothing_length/smoothing_length/smoothing_length;
    float m = 1.0f; //mass = 1 ??


    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        float4 r = p - pos[j];
        float rlen = magnitude(r);
        if(rlen < smoothing_length)
        {
            float r2 = rlen*rlen;
            float re2 = smoothing_length*smoothing_length;
            if(r2/re2 <= 4.f)
            {
                float R = sqrt(r2/re2);
                float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                float kern = m * Wij * (Pi + Pj) / (density[i] * density[j]);
                force[i] += kern * r;
            }

        }
    }
 

}
);

