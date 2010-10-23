
//----------------------------------------------------------------------
__kernel void subtract(int num, 
						__global int* cell_indices_end, 
                       __global int* cell_indices_start, 
                       __global int* cell_indices_nb) 
{
	uint gid = get_global_id(0);
	if (gid >= num) return;

	int start = cell_indices_start[gid];
	int nb = cell_indices_nb[gid];

	if (start < 0) {
		nb = 0;
	} else {
		nb = cell_indices_end[gid] - start;
	}

	cell_indices_nb[gid] = nb;
}
//----------------------------------------------------------------------
