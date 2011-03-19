#ifndef RTPS_FORCEFIELD_H_INCLUDED
#define RTPS_FORCEFIELD_H_INCLUDED

namespace rtps
{

enum FFType{ATTRACTOR, REPELER};

//keep track of the fluid settings
#ifdef WIN32
#pragma pack(2)
#endif
typedef struct ForceField
{
    float4 center;
    float radius;
    float max_force;
    float f;        //memory padding for opencl
    float ff;
    //FFType type;
    //unsigned int type;
    //unsigned int padd;

    ForceField(){};
    //ForceField(float4 center, float radius, float max_force, unsigned int type, unsigned int padd)
    ForceField(float4 center, float radius, float max_force)
    {
        this->center = center;
        this->radius = radius;
        this->max_force = max_force;
        this->f = 0;
        this->ff = 0;
        //this->type = type;
        //this->padd = padd;
    }

} ForceField 
#ifndef WIN32
__attribute__((aligned(16)));
#else
;
#endif



}


#endif
