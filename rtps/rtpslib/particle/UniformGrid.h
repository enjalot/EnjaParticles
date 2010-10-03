#ifndef UNIFORMGRID_H_INCLUDED
#define UNIFORMGRID_H_INCLUDED

#include "../structs.h"

namespace rtps {

class UniformGrid
{
public:
    UniformGrid(){};
    UniformGrid(float4 min, float4 max, float cell_size);
	UniformGrid(float4 min, float4 max, int nb_cells_x);
    ~UniformGrid();

    void make_cube(float4 *positions, float spacing, int num);

    float4 getMin(){ return min; };
    float4 getMax(){ return max; };
	float4 getDelta() { return delta; };
	float4 getRes() { return res; };
	float4 getSize() { return size; };

private:
    float4 min;
    float4 max; 

    float4 size;
    float4 res;
    float4 delta;


};
   
}
#endif
