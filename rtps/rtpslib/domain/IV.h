//Initial Value functions
#include "../structs.h"
#include <vector>
using namespace std;

namespace rtps
{


vector<float4> addRect(int num, float4 min, float4 max, float spacing);
vector<float4> addSphere(int num, float4 center, float radius, float spacing);
//spray [rate] particles per update until [num] particles have been sprayed
//vector<float3> addHose(int num, float3 origin, float3 normal, int rate);



}
