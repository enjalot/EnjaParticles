#include "structs.h"
#include <math.h>

namespace rtps
{

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}
float dot(float4 a, float4 b)
{
        return a.x*b.x + a.y*b.y + a.z*b.z;
}

float4 normalize(float4 a)
{
    float mag = magnitude(a);
    return float4(a.x/mag, a.y/mag, a.z/mag, a.w);
}

float4 cross(float4 a, float4 b)
{
    return float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

}
