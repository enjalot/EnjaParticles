#ifndef IV_H_INCLUDE
#define IV_H_INCLUDE

//Initial Value functions
#include "../structs.h"
#include <vector>
#include "../rtps_common.h"
using namespace std;

namespace rtps
{
    RTPS_EXPORT vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale);
    RTPS_EXPORT vector<float4> addSphere(int num, float4 center, float radius, float spacing, float scale);
    RTPS_EXPORT std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);
    RTPS_EXPORT std::vector<float4> addDiscRandom(int num, float4 center, float4 u, float4 v, float radius, float spacing);
}

#endif
