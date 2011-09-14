/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include <OUTER.h>
#include <math.h>

namespace rtps
{
namespace outer
{
    //----------------------------------------------------------------------
    Density::Density(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
     
        printf("load density\n");

        try
        {
            path = path + "/density.cl";
            k_density = Kernel(cli, path, "density_update");
        }
        catch (cl::Error er)
        {
            printf("ERROR(Density): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }


    }
    //----------------------------------------------------------------------

    void Density::execute(int num,
                    //input
                    //Buffer<float4>& svars, 
                    Buffer<float4>& pos_s,
                    Buffer<float>& dens_s,
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<OUTERParams>& sphp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    { 
        int iarg = 0;
        //k_density.setArg(iarg++, svars.getDevicePtr());
        k_density.setArg(iarg++, pos_s.getDevicePtr());
        k_density.setArg(iarg++, dens_s.getDevicePtr());
        k_density.setArg(iarg++, ci_start.getDevicePtr());
        k_density.setArg(iarg++, ci_end.getDevicePtr());
        k_density.setArg(iarg++, gp.getDevicePtr());
        k_density.setArg(iarg++, sphp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_density.setArg(iarg++, clf_debug.getDevicePtr());
        k_density.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_density.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(density): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

#if 0 //printouts    
        //DEBUGING
        
        if(num > 0)// && choice == 0)
        {
            printf("============================================\n");
            printf("which == %d *** \n", choice);
            printf("***** PRINT neighbors diagnostics ******\n");
            printf("num %d\n", num);

            std::vector<int4> cli(num);
            std::vector<float4> clf(num);
            
            cli_debug.copyToHost(cli);
            clf_debug.copyToHost(clf);

            std::vector<float4> poss(num);
            std::vector<float4> dens(num);

            for (int i=0; i < num; i++)
            //for (int i=0; i < 10; i++) 
            {
                //printf("-----\n");
                printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                //if(clf[i].w == 0.0) exit(0);
                //printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
                //		printf("pos : %f, %f, %f, %f\n", pos[i].x, pos[i].y, pos[i].z, pos[i].w);
            }
        }
#endif

    }

} // namespace outer







    float OUTER::Wpoly6(float4 r, float h)
    {
        float h9 = h*h*h * h*h*h * h*h*h;
        float alpha = 315.f/64.0f/sphp.PI/h9;
        float r2 = dist_squared(r);
        float hr2 = (h*h - r2);
        float Wij = alpha * hr2*hr2*hr2;
        return Wij;
    }


    void OUTER::cpuDensity()
    {
		#if 0
        float h = sphp.smoothing_distance;
        //stuff from Tim's code (need to match #s to papers)
        //float alpha = 315.f/208.f/params.PI/h/h/h;
        //
        //float h9 = h*h*h * h*h*h * h*h*h;
        //float alpha = 315.f/64.0f/params.PI/h9;
        //printf("alpha: %f\n", alpha);

        //sooo slow t.t

        float scale = sphp.simulation_scale;
        float sum_densities = 0.0f;

        for (int i = 0; i < num; i++)
        {
            float4 p = positions[i];
            p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);
            densities[i] = 0.0f;

            int neighbor_count = 0;
            for (int j = 0; j < num; j++)
            {
                if (j == i) continue;
                float4 pj = positions[j];
                pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
                float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);
                //error[i] = r;
                float rlen = magnitude(r);
                if (rlen < h)
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
                        densities[i] += sphp.mass * Wij;
                    }
                }

            }
            //printf("neighbor_count[%d] = %d; density = %f\n", i, neighbor_count, densities[i]);
            //sum_densities += densities[i];
        }
        //printf("CPU: sum_densities = %f\n", sum_densities);
		#endif
    }

}
