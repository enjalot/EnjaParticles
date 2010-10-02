#include "../GE_SPH.h"
#include "timer_macros.h"

namespace rtps {

void GE_SPH::computeEuler()
{
	static bool first_time = true;

	ts_cl[TI_EULER]->start(); // OK

	if (first_time) {
		try {
			std::string path(CL_SPH_UTIL_SOURCE_DIR);
			printf("path= %s\n", path.c_str());
			path = path + "/euler_cl.cl";
			printf("path= %s\n", path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	k_euler = Kernel(ps->cli, strg, "ge_euler");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(computeEuler): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = k_euler;
	int workSize = 128;

    kern.setArg(0, cl_sort_indices->getDevicePtr());
    kern.setArg(1, cl_vars_unsorted->getDevicePtr());
    kern.setArg(2, cl_vars_sorted->getDevicePtr());
    kern.setArg(3, cl_params->getDevicePtr());
    kern.setArg(4, ps->settings.dt); //time step

//	printf("dt= %f\n", ps->settings.dt);

	cl_vars_unsorted->copyToHost();
	float4* vars = cl_vars_unsorted->getHostPtr();
	printf("==================\n");
	for (int i=0; i < 5; i++) {
		int i1 = i + 1*nb_el;
		int i2 = i + 2*nb_el;
		printf("pos[%d] = %f, %f, %f, %f, %f\n", i, vars[i1].x, vars[i1].y, vars[i1].z, vars[i1].w);
		printf("vel[%d] = %f, %f, %f, %f, %f\n", i, vars[i2].x, vars[i2].y, vars[i2].z, vars[i2].w);
	}
	

   	kern.execute(nb_el, workSize); 

    ps->cli->queue.finish();
	ts_cl[TI_EULER]->end(); // OK
} 

#if 0
void GE_SPH::cpuEuler()
{
    float h = ps->settings.dt;
    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];

        //external force is gravity
        f.z += -9.8f;

        float speed = magnitude(f);
        if(speed > 600.0f) //velocity limit, need to pass in as struct
        {
            f.x *= 600.0f/speed;
            f.y *= 600.0f/speed;
            f.z *= 600.0f/speed;
        }

        float scale = params.simulation_scale;
        v.x += h*f.x / scale;
        v.y += h*f.y / scale;
        v.z += h*f.z / scale;
        
        p.x += h*v.x;
        p.y += h*v.y;
        p.z += h*v.z;
        p.w = 1.0f; //just in case

        velocities[i] = v;
        positions[i] = p;
    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}
#endif

}
