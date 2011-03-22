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

void addCube(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& rvec);

//spray [rate] particles per update until [num] particles have been sprayed
//vector<float3> addHose(int num, float3 origin, float3 normal, int rate);


vector<float4> addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax);
vector<float4> addRandArrangement(int num, float scale, float4 dmin, float4 dmax);
vector<float4> addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax);
}

#endif
