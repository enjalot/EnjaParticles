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
std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);



}

#endif
