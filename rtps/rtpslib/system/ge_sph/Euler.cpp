#include "../GE_SPH.h"

namespace rtps {

//----------------------------------------------------------------------
void GE_SPH::computeEuler()
{
	static bool first_time = true;

	ts_cl[TI_EULER]->start(); // OK

	if (first_time) {
		try {
			std::string path(CL_SPH_SOURCE_DIR);
			printf("path= %s\n", path.c_str());
			path = path + "/euler.cl";
			printf("path= %s\n", path.c_str());
			int length;
			const char* src = file_contents(path.c_str(), &length);
			std::string strg(src);
        	k_euler = Kernel(ps->cli, strg, "ge_euler");
			first_time = false;
		} catch(cl::Error er) {
        	printf("ERROR(buildDataStructures): %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(1);
		}
	}

    Kernel kern = k_euler;
  
    //TODO: fix the way we are wrapping buffers
    kern.setArg(0, nb_vars);
    kern.setArg(1, cl_vars_sorted->getDevicePtr());
    kern.setArg(2, ps->settings.dt); //time step

	int workSize = 128;
   	kern.execute(nb_el, workSize); 

	ps->cli->queue.finish();
	ts_cl[TI_EULER]->end(); // OK

} 
//----------------------------------------------------------------------
#if 0
void GE_SPH::loadEuler()
{
    #include "euler.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_euler = Kernel(ps->cli, euler_program_source, "euler");
  
    //TODO: fix the way we are wrapping buffers
    k_euler.setArg(0, cl_position.cl_buffer[0]);
    k_euler.setArg(1, cl_velocity.cl_buffer[0]);
    k_euler.setArg(2, cl_force.cl_buffer[0]);
    k_euler.setArg(3, ps->settings.dt); //time step

} 
#endif
//----------------------------------------------------------------------

}
