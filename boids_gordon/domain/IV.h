#ifndef IV_H_INCLUDE
#define IV_H_INCLUDE

//Initial Value functions
#include "../structs.h"
#include <vector>
using namespace std;

namespace rtps
{


vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale);
vector<float4> addSphere(int num, float4 center, float radius, float spacing, float scale);
vector<float4> addCircle(int num, float4 center, float radius, float spacing, float scale);

//spray [rate] particles per update until [num] particles have been sprayed
//vector<float3> addHose(int num, float3 origin, float3 normal, int rate);

std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);

vector<float4> addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax);
vector<float4> addRandArrangement(int num, float scale, float4 dmin, float4 dmax);
vector<float4> addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax);
}

#endif
