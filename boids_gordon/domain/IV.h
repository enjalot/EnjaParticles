#ifndef IV_H_INCLUDE
#define IV_H_INCLUDE

//Initial Value functions
#include "../structs.h"
#include <vector>
using namespace std;

namespace rtps
{


void GE_addRect(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& output);
void addRect(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& output);
void addSphere(int num, float4 center, float radius, float spacing, float scale, std::vector<float4>& output);
void addCircle(int num, float4 center, float radius, float spacing, float scale, std::vector<float4>& output);

//spray [rate] particles per update until [num] particles have been sprayed
//vector<float3> addHose(int num, float3 origin, float3 normal, int rate);

void addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing, std::vector<float4>& output);

void addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
void addRandArrangement(int num, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
void addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
}

#endif
