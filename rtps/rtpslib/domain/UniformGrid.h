#ifndef UNIFORMGRID_H_INCLUDED
#define UNIFORMGRID_H_INCLUDED

#include "../structs.h"

namespace rtps {

class UniformGrid
{
public:
    UniformGrid(){};
    UniformGrid(float4 min, float4 max, float cell_size);
    ~UniformGrid();

    void make_cube(float4 *positions, float spacing, int num);
    void make_column(float4 *positions, float spacing, int num);
    void make_dam(float4 *positions, float spacing, int num);
    int make_line(float4 *positions, float spacing, int num);

    float4 getMin(){ return min; };
    float4 getMax(){ return max; };

private:
    float4 min;
    float4 max; 

    float4 size;
    float4 res;
    float4 delta;


};
   
}
#endif
