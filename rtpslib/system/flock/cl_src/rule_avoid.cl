#ifndef _RULE_AVOID_CL_
#define _RULE_AVOID_CL_

float4 position_t = target * flockp->simulation_scale;
position_t.w = 0.f;

float4 dist = normalize(position_t - position_i);
float4 desiredVel = dist * flockp->max_speed;

pt.avoid = (-desiredVel - velocity_i);
pt.avoid.w = 0.f;

#endif
