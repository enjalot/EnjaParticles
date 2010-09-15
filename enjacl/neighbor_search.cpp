
#include <array_opencl_1d.h>
using namespace std;

#include <CL/cl_platform.h>
#include <CL/cl.h>

//----------------------------------------------------------------------
char* EnjaParticles::getSourceString(const char* path_to_source_file)
{
	// Must find a way to only compile a single time. 
	// Define all programs before starting the code? 

	//printf("enter addProgramR\n");

    FILE* fd =  fopen(path_to_source_file, "r");
	if (fd == 0) {
		printf("cannot open file: %s\n", path_to_source_file);
	}
// should not limit string size
	int max_len = 300000;
    char* source = new char [max_len];
    int nb = fread(source, 1, max_len, fd);    

	if (nb > (max_len-2)) { 
        printf("cannot read program from %s\n", path_to_source_file);
        printf("   buffer size too small\n");
    }    
	source[nb] = '\0';

	return source;
}
//----------------------------------------------------------------------
void EnjaParticles::neighbor_search()
{
	static cll_Program* prog = 0;

	if (prog == 0) {
		try {
			string path(CL_SOURCE_DIR);
			path = path + "/uniform_grid_utils.cl";
			char* src = getSourceString(path.c_str());
        	step1_program = loadProgram(src);
        	step1_kernel = cl::Kernel(step1_program, "K_SumStep1", &err);
		} catch(cl::Error er) {
        	printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
		}
	}

	cl::Kernel kern = step1_kernel;
	printf("sizeof(kern) = %d\n", sizeof(kern));


	int iarg = 0;

	kern.setArg(iarg++, nb_el);
	kern.setArg(iarg++, nb_vars);

	kern.setArg(iarg++, cl_vars_unsorted);
	kern.setArg(iarg++, cl_vars_sorted);
	kern.setArg(iarg++, cl_cell_indices_start);
	kern.setArg(iarg++, cl_cell_indices_end);
	kern.setArg(iarg++, cl_GridParams);


	size_t global = (size_t) nb_el;
	//size_t local = cl.getMaxWorkSize(kern.getKernel());
	//size_t local = cl.getMaxWorkSize(kern());
	//printf("local= %d, global= %d\n", local, global);
	int local = 128;

    err = queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(nb_el), cl::NDRange(local), NULL, &event);

	queue.finish();
	printf("after end of neighbor_search\n");

	//cl_event exec = kern.exec(1, &global, &local);
	//cl.waitForKernelsToFinish();
}

