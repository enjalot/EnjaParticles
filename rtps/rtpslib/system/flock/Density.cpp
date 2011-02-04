#include "../FLOCK.h"
#include <math.h>

/*
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}
*/



namespace rtps {

void FLOCK::loadDensity()
{
    printf("create density kernel\n");

    std::string path(FLOCK_CL_SOURCE_DIR);
    path += "/density_cl.cl";
    k_density = Kernel(ps->cli, path, "density");
  
    k_density.setArg(0, cl_position.getDevicePtr());
    k_density.setArg(1, cl_density.getDevicePtr());
    k_density.setArg(2, cl_FLOCKParams.getDevicePtr());
    //k_density.setArg(3, cl_error_check.cl_buffer[0]);

} 

float FLOCK::Wpoly6(float4 r, float h)
{
    float h9 = h*h*h * h*h*h * h*h*h;
    float alpha = 315.f/64.0f/params.PI/h9;
    float r2 = dist_squared(r);
    float hr2 = (h*h - r2);
    float Wij = alpha * hr2*hr2*hr2;
    return Wij;
}


void FLOCK::cpuDensity()
{
    float h = params.smoothing_distance;
    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params.PI/h/h/h;
    //
    //float h9 = h*h*h * h*h*h * h*h*h;
    //float alpha = 315.f/64.0f/params.PI/h9;
    //printf("alpha: %f\n", alpha);

    //sooo slow t.t

    float scale = params.simulation_scale;
    float sum_densities = 0.0f;

    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);
        densities[i] = 0.0f;

        int neighbor_count = 0;
        for(int j = 0; j < num; j++)
        {
            if(j == i) continue;
            float4 pj = positions[j];
            pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
            float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);
            //error[i] = r;
            float rlen = magnitude(r);
            if(rlen < h)
            {
                float r2 = dist_squared(r);
                float re2 = h*h;
                //if(r2/re2 <= 4.f)
                //if(r/h <= 2.f)
                {
                    //printf("i: %d j: %d\n", i, j);
                    neighbor_count++;
                    //float R = sqrt(r2/re2);
                    //float Wij = alpha*(2.f/3.f - 9.f*R*R/8.f + 19.f*R*R*R/24.f - 5.f*R*R*R*R/32.f);
                    //
                    //float hr2 = (h*h - r2);
                    //float Wij = alpha * hr2*hr2*hr2;
                    float Wij = Wpoly6(r, h);
                    /*
                    if(i == j)
                    {
                        printf("rlen: %f\n", rlen);
                        printf("Wij = %f\n", Wij);
                    }
                    */
                   //printf("%f ", Wij);
                    densities[i] += params.mass * Wij;
                }
            }
     
        }
        //printf("neighbor_count[%d] = %d; density = %f\n", i, neighbor_count, densities[i]);
        //sum_densities += densities[i];
    }
    //printf("CPU: sum_densities = %f\n", sum_densities);
}

}
