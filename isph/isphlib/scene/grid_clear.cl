__kernel void ClearCellStart(__global uint* cellsStart : CELLS_START)
{
	cellsStart[get_global_id(0)] = UINT_MAX;
}
