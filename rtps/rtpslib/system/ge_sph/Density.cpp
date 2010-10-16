#include "../GE_SPH.h"
#include <math.h>
#include "timer_macros.h"
#include <string>

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::computeDensity()
{
#if 1
	static bool first_time = true;

	ts_cl[TI_DENS]->start(); // OK

	if (first_time) {
		try {
			std::string path(CL_SPH_SOURCE_DIR);
			printf("path= %s\n", path.c_str());
			path = path + "/density.cl";
			printf("path= %s\n", path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	k_density = Kernel(ps->cli, strg, "ge_density");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = k_density;
	int workSize = 128;

    k_density.setArg(0, nb_vars);
    k_density.setArg(1, cl_vars_sorted->getDevicePtr());
    k_density.setArg(2, cl_params->getDevicePtr());
    k_density.setArg(3, cl_error_check->getDevicePtr());

   	kern.execute(nb_el, workSize); 

	//printBuildDiagnostics();

    ps->cli->queue.finish();
	ts_cl[TI_DENS]->end();
#endif
}
//----------------------------------------------------------------------
#if 0
void GE_SPH::loadDensity()
{
    #include "density.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_density = Kernel(ps->cli, density_program_source, "density");
  
    //TODO: fix the way we are wrapping buffers
    k_density.setArg(0, cl_position.cl_buffer[0]);
    k_density.setArg(1, cl_density.cl_buffer[0]);
    k_density.setArg(2, cl_params.cl_buffer[0]);
    k_density.setArg(3, cl_error_check.cl_buffer[0]);
} 
#endif

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

//----------------------------------------------------------------------
void GE_SPH::cpuDensity()
{
	GE_SPHParams& params = *(cl_params->getHostPtr());
    float h = params.smoothing_distance;
    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params.PI/h/h/h;
    float h9 = h*h*h * h*h*h * h*h*h;
    float alpha = 315.f/64.0f/params.PI/h9;
    printf("alpha: %f\n", alpha);

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
                if(r2/re2 <= 4.f)
                {
                    //printf("i: %d j: %d\n", i, j);
                    neighbor_count++;
                    //float R = sqrt(r2/re2);
                    //float Wij = alpha*(2.f/3.f - 9.f*R*R/8.f + 19.f*R*R*R/24.f - 5.f*R*R*R*R/32.f);
                    float hr2 = (h*h - r2);
                    float Wij = alpha * hr2*hr2*hr2;
                    printf("%f ", Wij);
                    densities[i] += params.mass * Wij;
                }
            }
     
        }
        printf("neighbor_count[%d] = %d; density = %f\n", i, neighbor_count, densities[i]);
        sum_densities += densities[i];
    }
    printf("CPU: sum_densities = %f\n", sum_densities);
}
//----------------------------------------------------------------------

}
