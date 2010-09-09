#ifndef UNIFORMGRID_H_INCLUDED
#define UNIFORMGRID_H_INCLUDED


#include "enja.h"

typedef struct UniformGrid::Params
{
    float3 min;
    float3 max; 

    float3 size;
    float3 res;
    float3 delta;
} Params;

class UniformGrid
{
    Params params;    
    UniformGrid();

};

#endif
