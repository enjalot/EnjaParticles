#include "../SPH.h"

namespace rtps {

void SPH::loadXSPH()
{
    #include "viscosity.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_viscosity = Kernel(ps->cli, viscosity_program_source, "viscosity");
  
    //TODO: fix the way we are wrapping buffers
    k_viscosity.setArg(0, cl_position.cl_buffer[0]);
    k_viscosity.setArg(1, cl_velocity.cl_buffer[0]);
    k_viscosity.setArg(2, cl_density.cl_buffer[0]);
    k_viscosity.setArg(3, cl_force.cl_buffer[0]);
    k_viscosity.setArg(4, cl_params.cl_buffer[0]);

} 


void SPH::cpuXSPH()
{

    float scale = params.simulation_scale;
    float h = params.smoothing_distance;

    for(int i = 0; i < num; i++)
    {

        float4 p = positions[i];
        float4 v = velocities[i];
        p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);
        v = float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);

        float4 f = float4(0.0f, 0.0f, 0.0f, 0.0f);

        //stuff from Tim's code (need to match #s to papers)
        //float alpha = 315.f/208.f/params->PI/h/h/h;

        for(int j = 0; j < num; j++)
        {
            if(j == i) continue;
            float4 pj = positions[j];
            float4 vj = velocities[j];
            pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
            vj = float4(vj.x * scale, vj.y * scale, vj.z * scale, vj.w * scale);
            float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);
 
            float rlen = magnitude(r);
            if(rlen < h)
            {
                float r2 = rlen*rlen;
                float re2 = h*h;
                if(r2/re2 <= 4.f)
                {
                    //float R = sqrt(r2/re2);
                    //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                    float Wij = Wpoly6(r, h);
                    float fcoeff = 2.0f * params.mass * Wij / (densities[j] + densities[i]);
                    f.x += fcoeff * (velocities[j].x - v.x); 
                    f.y += fcoeff * (velocities[j].y - v.y); 
                    f.z += fcoeff * (velocities[j].z - v.z); 
                }

            }
        }
        //modifies velocity 
        /*
        forces[i].x += f.x;
        forces[i].y += f.y;
        forces[i].z += f.z;
        */
    }

}


}
