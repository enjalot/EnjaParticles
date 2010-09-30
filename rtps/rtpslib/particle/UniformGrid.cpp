#include <math.h>

#include "UniformGrid.h"

namespace rtps {

UniformGrid::UniformGrid(float3 min, float3 max, float cell_size)
{
    this->min = min;
    this->max = max;
    size = float3(max.x - min.x,
                  max.y - min.y,
                  max.z - min.z);

    res = float3(ceil(size.x / cell_size),
                 ceil(size.y / cell_size),
                 ceil(size.z / cell_size));

    size = float3(res.x * cell_size,
                  res.y * cell_size,
                  res.z * cell_size);

    delta = float3(res.x / size.x,
                   res.y / size.y,
                   res.z / size.z);

}

UniformGrid::~UniformGrid()
{
}

void UniformGrid::make_cube(float4* position, float spacing, int num)
{
    //float xmin = min.x/2.5f;
    float xmin = min.x + max.x/2.5f;
    float xmax = max.x/4.0f;
    //float ymin = min.y;
    float ymin = min.y + max.y/4.0f;
    float ymax = max.y;
    //float zmin = min.z/2.0f;
    float zmin = min.z;// + max.z/2.0f;
    float zmax = max.z/4.5f;

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

int UniformGrid::make_line(float4* position, float spacing, int num)
{
    float xmin = min.x;
    float xmax = max.x;
    int i = 0;
    float y = min.y + (max.y - min.y)/2.0f;
    float z = min.z + (max.z - min.z)/2.0f;
    for (float x = xmin; x <= xmax; x+= spacing) {
        if(i >= num) break;
        position[i] = float4(x,y,z,1.0f);
        i++;
    }
    return i;
}


}
