#if DIM == 3


uint4 CellPos(vector pos, vector gridStart, scalar cellSizeInv)
{
	vector gridPos = floor((pos - gridStart) * cellSizeInv);
    return convert_uint4(gridPos);
}

uint CellHash(uint4 cell, uint4 cellCount)
{
	return (cell.x*cellCount.x + cell.y) * cellCount.y + cell.z;
}

#define ForEachSetup(POS,GRID_START,CELLSIZE_INV,CELL_COUNT) \
	uint4 _cellI = CellPos(POS, GRID_START, CELLSIZE_INV); \
	uint4 _loopStart = _cellI - (uint4)1u; \
	uint4 _loopEnd = min(_cellI+(uint4)1u, CELL_COUNT-(uint4)1u); \
	uint4 _cellJ;

#define ForEachNeighbor(CELL_COUNT,CELLS_HASH,HASHES_PARTICLE,CELLS_START) \
	for(_cellJ.x=_loopStart.x; _cellJ.x<=_loopEnd.x; _cellJ.x++) \
	for(_cellJ.y=_loopStart.y; _cellJ.y<=_loopEnd.y; _cellJ.y++) \
	for(_cellJ.z=_loopStart.z; _cellJ.z<=_loopEnd.z; _cellJ.z++){ \
		uint _hash = CellHash(_cellJ, CELL_COUNT); \
		uint _j = CELLS_START[_hash]; \
		if(_j == UINT_MAX) continue; \
		for(uint particleJ=CELLS_HASH[_j]; _hash==particleJ; particleJ=CELLS_HASH[++_j]){ \
			uint j = HASHES_PARTICLE[_j]; \
			if(j==i) continue;

#else


uint2 CellPos(vector pos, vector gridStart, scalar cellSizeInv)
{
	//return convert_uint2(floor(pos - gridStart) * cellSizeInv);
	return (uint2)((uint)floor((pos.x-gridStart.x)*cellSizeInv), (uint)floor((pos.y-gridStart.y)*cellSizeInv));
}

uint CellHash(uint2 cell, uint2 cellCount)
{
	//cell = min(cell, cellCount-(uint2)1u);
	if(cell.x >= cellCount.x || cell.y >= cellCount.y)
		return UINT_MAX;
	return cell.y * cellCount.x + cell.x;
}

#define ForEachSetup(POS,GRID_START,CELLSIZE_INV,CELL_COUNT) \
	uint2 _cellI = CellPos(POS, GRID_START, CELLSIZE_INV); \
	uint2 _loopStart = (uint2)(_cellI.x>0?_cellI.x-1:0, _cellI.y>0?_cellI.y-1:0); \
	uint2 _loopEnd = (uint2)(min(_cellI.x+1,CELL_COUNT.x-1), min(_cellI.y+1,CELL_COUNT.y-1)); \
	uint2 _cellJ;

#define ForEachNeighbor(CELL_COUNT,CELLS_HASH,HASHES_PARTICLE,CELLS_START) \
	for(_cellJ.x=_loopStart.x; _cellJ.x<=_loopEnd.x; _cellJ.x++) \
	for(_cellJ.y=_loopStart.y; _cellJ.y<=_loopEnd.y; _cellJ.y++){ \
		uint _hash = CellHash(_cellJ, CELL_COUNT); \
		uint _j = CELLS_START[_hash]; \
		if(_j == UINT_MAX) continue; \
		for(uint particleJ=CELLS_HASH[_j]; _hash==particleJ; particleJ=CELLS_HASH[++_j]){ \
			uint j = HASHES_PARTICLE[_j]; \
			if(j==i) continue;

#endif


#define ForEachEnd }}
