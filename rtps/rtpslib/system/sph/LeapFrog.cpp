#include "../SPH.h"

namespace rtps
{

    void SPH::loadLeapFrog()
    {
        printf("create leapfrog kernel\n");

        std::string path(SPH_CL_SOURCE_DIR);
        path += "/leapfrog.cl";
        k_leapfrog = Kernel(ps->cli, path, "leapfrog");

        /*
        //TODO: fix the way we are wrapping buffers
        k_leapfrog.setArg(0, cl_position.getDevicePtr());
        k_leapfrog.setArg(1, cl_velocity.getDevicePtr());
        k_leapfrog.setArg(2, cl_veleval.getDevicePtr());
        k_leapfrog.setArg(3, cl_force.getDevicePtr());
        k_leapfrog.setArg(4, cl_xsph.getDevicePtr());
        k_leapfrog.setArg(5, cl_color.getDevicePtr());
        k_leapfrog.setArg(6, ps->settings.dt); //time step
        k_leapfrog.setArg(7, cl_SPHParams.getDevicePtr());
        */
    } 
    void SPH::leapfrog()
    {

        int iargs = 0;
        k_leapfrog.setArg(iargs++, cl_sort_indices.getDevicePtr());
        k_leapfrog.setArg(iargs++, cl_vars_unsorted.getDevicePtr());
        k_leapfrog.setArg(iargs++, cl_vars_sorted.getDevicePtr());
        k_leapfrog.setArg(iargs++, cl_position.getDevicePtr());
        //    k_leapfrog.setArg(iargs++, cl_color.getDevicePtr());
        k_leapfrog.setArg(iargs++, cl_sphp.getDevicePtr());
        k_leapfrog.setArg(iargs++, ps->settings.dt); //time step

        int local_size = 128;
        k_leapfrog.execute(num, local_size);


#if 0
#define DENS 0
#define POS 1
#define VEL 2

        printf("************ LeapFrog **************\n");
            int nbc = num+5;
            std::vector<float4> poss(nbc);
            std::vector<float4> uposs(nbc);
            std::vector<float4> dens(nbc);

            //cl_vars_sorted.copyToHost(dens, DENS*sphp.max_num);
            cl_vars_sorted.copyToHost(poss, POS*sphp.max_num);
            cl_vars_unsorted.copyToHost(uposs, POS*sphp.max_num);

            for (int i=0; i < nbc; i++)
            //for (int i=0; i < 10; i++) 
            {
                poss[i] = poss[i] / sphp.simulation_scale;
                //printf("-----\n");
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                printf("pos sorted: %f, %f, %f, %f\n", poss[i].x, poss[i].y, poss[i].z, poss[i].w);
                printf("pos unsorted: %f, %f, %f, %f\n", uposs[i].x, uposs[i].y, uposs[i].z, uposs[i].w);
                //printf("dens sorted: %f, %f, %f, %f\n", dens[i].x, dens[i].y, dens[i].z, dens[i].w);
            }

#endif

        /*
         * enables us to cut off after a couple iterations
         * by setting cut = 1 from some other function
        if(cut >= 1)
        {
            if (cut == 2) {exit(0);}
            cut++;
        }
        */


    }

    void SPH::cpuLeapFrog()
    {
        float h = ps->settings.dt;
        for (int i = 0; i < num; i++)
        {
            float4 p = positions[i];
            float4 v = velocities[i];
            float4 f = forces[i];

            //external force is gravity
            f.z += -9.8f;

            float speed = magnitude(f);
            if (speed > 600.0f) //velocity limit, need to pass in as struct
            {
                f.x *= 600.0f/speed;
                f.y *= 600.0f/speed;
                f.z *= 600.0f/speed;
            }

            float4 vnext = v;
            vnext.x += h*f.x;
            vnext.y += h*f.y;
            vnext.z += h*f.z;

            float xsphfactor = .1f;
            vnext.x += xsphfactor * xsphs[i].x;
            vnext.y += xsphfactor * xsphs[i].y;
            vnext.z += xsphfactor * xsphs[i].z;

            float scale = sphp.simulation_scale;
            p.x += h*vnext.x / scale;
            p.y += h*vnext.y / scale;
            p.z += h*vnext.z / scale;
            p.w = 1.0f; //just in case

            veleval[i].x = (v.x + vnext.x) *.5f;
            veleval[i].y = (v.y + vnext.y) *.5f;
            veleval[i].z = (v.z + vnext.z) *.5f;

            velocities[i] = vnext;
            positions[i] = p;

        }
        //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
    }

}
