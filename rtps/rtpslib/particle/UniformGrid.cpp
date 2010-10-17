#include <math.h>
#include <stdlib.h>

#include "UniformGrid.h"

namespace rtps {

//----------------------------------------------------------------------
#if 0
UniformGrid::UniformGrid(float4 min, float4 max, float cell_size)
{
    this->min = min;    
	this->max = max;    
	size = float4(max.x - min.x, 
				  max.y - min.y, 
				  max.z - min.z);
    res = float4(ceil(size.x / cell_size),
                 ceil(size.y / cell_size),
				 ceil(size.z / cell_size));
    size = float4(res.x * cell_size,
                  res.y * cell_size,
				  res.z * cell_size);
    delta = float4(res.x / size.x,
				   res.y / size.y,
                   res.z / size.z);
}
#endif
//----------------------------------------------------------------------
UniformGrid::UniformGrid(float4 min, float4 max, float cell_size, float sim_scale)
{
	this->bnd_min = min;
	this->bnd_max = max;

	double s2 = 2.*cell_size;
	min = min - float4(s2, s2, s2, 0.);
	max = max + float4(s2, s2, s2, 0.);

	this->sim_scale = sim_scale;
    this->min = min;
    this->max = max;


	float world_cell_size = cell_size / sim_scale;

	// domain dimensions in world coordinates (that are plotted)
    size = float4(max.x - min.x,
                  max.y - min.y,
                  max.z - min.z, 1.);

	//  how many cells in each direction. 
    res = float4(ceil(size.x / world_cell_size),
                 ceil(size.y / world_cell_size),
                 ceil(size.z / world_cell_size), 1.);

	// Modified dimensions
    size = float4(res.x * world_cell_size,
                  res.y * world_cell_size,
                  res.z * world_cell_size,
				  1.);

	// width, height, depth of a single cell
    //delta = float4(res.x / size.x,
                   //res.y / size.y,
                   //res.z / size.z, 
				   //1.);

    delta = float4(size.x / res.x,    // res = nb cells
                   size.y / res.y,
                   size.z / res.z, 
				   1.);

}
//----------------------------------------------------------------------
UniformGrid::UniformGrid(float4 min, float4 max, int4 nb_cells, float sim_scale)
{
	this->sim_scale = sim_scale;
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
	printf("spacing (makeCube)= %f\n", spacing);
			return;
		}
        position[i] = float4(x,y,z,1.0f);
		i++;
		//printf("i= %d, pos= %f, %f, %f\n", x, y, z);
    }}}

	offset = i;
	//printf("makeCube, offset= %d, about to exit\n", offset); exit(0);
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
void UniformGrid::print()
{
	printf("\n--- Uniform Grid ---\n");
	printf("sim_scale= %f\n", sim_scale);
	min.print("min");
	max.print("max");
	size.print("size");
	res.print("res");
	delta.print("delta");
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
} // namespace
