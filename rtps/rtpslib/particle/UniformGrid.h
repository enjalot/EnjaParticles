#ifndef UNIFORMGRID_H_INCLUDED
#define UNIFORMGRID_H_INCLUDED

#include "../structs.h"

namespace rtps {

class UniformGrid
{
public:
    UniformGrid(){};
    UniformGrid(float3 min, float3 max, float cell_size);
    ~UniformGrid();

    void make_cube(float4 *positions, float spacing, int num);
    int make_line(float4 *positions, float spacing, int num);

    float3 getMin(){ return min; };
    float3 getMax(){ return max; };

private:
    float3 min;
    float3 max; 

    float3 size;
    float3 res;
    float3 delta;


};
   
}
#endif
