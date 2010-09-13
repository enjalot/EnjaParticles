
// I should not have to do this
#include "cl.h"

#include <string>
using namespace std;

#include "cll_kernel.h"

//----------------------------------------------------------------------
cll_Kernel::cll_Kernel(string name)
{
	this->name = name;
};
//----------------------------------------------------------------------
cl_event cll_Kernel::exec(cl_uint work_dim, const size_t *global_work_size, const size_t *local_work_size)
{
	cl_event event;


	// only works if single command queue per cl
    int err = clEnqueueNDRangeKernel(CL::commands, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &event);

    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        exit(1);
        //return EXIT_FAILURE;
    }

    return event;
}
//----------------------------------------------------------------------
