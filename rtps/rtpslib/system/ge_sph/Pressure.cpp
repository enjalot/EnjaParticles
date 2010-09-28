#include "../GE_SPH.h"

namespace rtps {

void GE_SPH::computePressure()
{
	static bool first_time = true;

	ts_cl[TI_PRES]->start(); // OK

	if (first_time) {
		try {
			std::string path(CL_SPH_SOURCE_DIR);
			printf("path= %s\n", path.c_str());
			path = path + "/pressure.cl";
			printf("path= %s\n", path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	k_pressure = Kernel(ps->cli, strg, "ge_pressure");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

	Kernel kern = k_pressure;
	int workSize = 128;

    kern.setArg(0, nb_vars);
    kern.setArg(1, cl_vars_sorted->getDevicePtr());
    kern.setArg(2, cl_params->getDevicePtr());

   	kern.execute(nb_el, workSize); 

	ps->cli->queue.finish();
	ts_cl[TI_PRES]->end(); // OK
} 
//----------------------------------------------------------------------
#if 0
void GE_SPH::loadPressure()
{
    #include "pressure.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_pressure = Kernel(ps->cli, pressure_program_source, "pressure");
  
    //TODO: fix the way we are wrapping buffers
    k_pressure.setArg(0, cl_position.cl_buffer[0]);
    k_pressure.setArg(1, cl_density.cl_buffer[0]);
    k_pressure.setArg(2, cl_force.cl_buffer[0]);
    k_pressure.setArg(3, cl_params.cl_buffer[0]);


} 
#endif


}
