#include <math.h>
#include <stdlib.h>

#include "UniformGrid.h"

namespace rtps {

//----------------------------------------------------------------------
UniformGrid::UniformGrid(float4 min, float4 max, float cell_size)
{
    this->min = min;
    this->max = max;

	// domain dimensions
    size = float4(max.x - min.x,
                  max.y - min.y,
                  max.z - min.z, 1.);

	//  how many cells in each direction. 
    res = float4(ceil(size.x / cell_size),
                 ceil(size.y / cell_size),
                 ceil(size.z / cell_size), 1.);

	// Modified dimensions
    size = float4(res.x * cell_size,
                  res.y * cell_size,
                  res.z * cell_size,
				  1.);

	// width, height, depth of a single cell
    //delta = float4(res.x / size.x,
                   //res.y / size.y,
                   //res.z / size.z, 
				   //1.);

    delta = float4(size.x / res.x, 
                   size.y / res.y,
                   size.z / res.z, 
				   1.);

}
//----------------------------------------------------------------------
UniformGrid::UniformGrid(float4 min, float4 max, int4 nb_cells)
{
    this->min = min;
    this->max = max;

	// domain dimensions
    size = float4(max.x - min.x,
                  max.y - min.y,
                  max.z - min.z, 1.);
	printf("UniformGrid: grid_size= %f, %f, %f\n", size.x, size.y, size.z);
	//exit(0);

	//  how many cells in each direction. 
    res = float4(nb_cells.x, 
    			 nb_cells.y, 
    			 nb_cells.z, 
    			 1);

    delta = float4(size.x / res.x, 
                   size.y / res.y,
                   size.z / res.z, 
				   1.);

}
//----------------------------------------------------------------------
UniformGrid::~UniformGrid()
{
}

//----------------------------------------------------------------------
void UniformGrid::make_cube(float4* position, float spacing, int num)
{
    //float xmin = min.x/2.5f;
    float xmin = min.x + max.x/2.5f;
    float xmax = max.x/4.0f;
    //float ymin = min.y;
    float ymin = min.y + max.y/4.0f;
    float ymax = max.y;
    //float zmin = min.z/2.0f;
    float zmin = min.z; // + max.z/2.0f;
    float zmax = min.z + max.z/3.0; //4.5f;
	printf("spacing= %f\n", spacing);

    int i=0;
    //cube in corner
    for (float y = ymin; y <= ymax; y+=spacing) {
    for (float z = zmin; z <= zmax; z+=spacing) {
    for (float x = xmin; x <= xmax; x+=spacing) {
        if (i >= num) break;				
        position[i] = float4(x,y,z,1.0f);
        i++;
    }}}

}

//----------------------------------------------------------------------
void UniformGrid::makeCube(float4* position, float4 pmin, float4 pmax, float spacing, int& num, int& offset)
{
    int i=offset;

    for (float z = pmin.z; z <= pmax.z; z+=spacing) {
    for (float y = pmin.y; y <= pmax.y; y+=spacing) {
    for (float x = pmin.x; x <= pmax.x; x+=spacing) {
        if (i >= num) {
			offset = num;
			return;
		}
        position[i] = float4(x,y,z,1.0f);
		i++;
    }}}

	offset = i;
}
//----------------------------------------------------------------------
void UniformGrid::makeSphere(float4* position, float4 center, float radius, int& num, int& offset, float spacing)
{
// offset: start counting particles from offset. Do not go beyond num
	
	int i=offset;
	float4 pmin;
	float4 pmax;

	pmin.x = center.x - radius;
	pmax.x = center.x + radius;
	pmin.y = center.y - radius;
	pmax.y = center.y + radius;
	pmin.z = center.z - radius;
	pmax.z = center.z + radius;

	// only fill bounding box of sphere
    for (float z = pmin.z; z <= pmax.z; z+=spacing) {
    for (float y = pmin.y; y <= pmax.y; y+=spacing) {
    for (float x = pmin.x; x <= pmax.x; x+=spacing) {
		if (i >= num) {
			offset = num;
			return;
		}
		float r2 = (x-center.x)*(x-center.x) + 
		         + (y-center.y)*(y-center.y) + 
		         + (z-center.z)*(z-center.z);
		if (r2 > (radius*radius)) continue;
		position[i] = float4(x,y,z,1.0);
		i++;
	}}}

	offset = i;

}
//----------------------------------------------------------------------
} // namespace
