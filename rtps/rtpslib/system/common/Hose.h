#ifndef HOSE_H_INCLUDED
#define HOSE_H_INCLUDED

#include "../../RTPS.h"
#include "../../structs.h"
#include <vector>
using namespace std;

namespace rtps
{

    
class Hose
{
public:
    Hose(RTPS *ps, int total_n, float4 center, float4 velocity, float radius, float spacing);
    //~Hose();

    void update(float4 center, float4 velocity, float radius, float spacing);
    std::vector<float4> spray();
    //refill();
    float4 getVelocity(){ return velocity;}


private:
    int total_n;        //total particles available to the hose
    int n_count;        //number of particles left in the hose

    float4 center;
    float4 velocity;
    float4 u, w;        //orthogonal vectors to velocity
    void calc_vectors();

    float radius;
    float spacing;

    void calc_em();     //calculate emission rate
    int em;             //how many calls to spray before emitting
    int em_count;       //what call we are on

    //we need the timestep and spacing from the settings
    RTPS *ps;

};

//std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);

//spray [rate] particles per update until [num] particles have been sprayed
//vector<float3> addHose(int num, float3 origin, float3 normal, int rate);



}

#endif
