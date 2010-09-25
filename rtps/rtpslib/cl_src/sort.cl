
#define STRINGIFY(A) #A

std::string sort_program_source = STRINGIFY(

__kernel void sort(__global int* sorted, __global int* unsorted)
{
	int id = get_global_id(0);
	sorted[id] = 2;
}
);
//----------------------------------------------------------------------
