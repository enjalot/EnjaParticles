#ifndef _CL_SIMPLE_STRUCTURES_H_
#define _CL_SIMPLE_STRUCTURES_H_

typedef struct ForceField
{
    float4 center;
    float radius;
    float max_force;
    float f;
    float ff;
    //unsigned int type;
    //unsigned int padd;

} ForceField;

#endif
