#ifndef _RULE_GOAL_CL_
#define _RULE_GOAL_CL_

float4 position_t = target * flockp->simulation_scale;
position_t.w = 0.f;

float4 dist = normalize(position_t - position_i);
float4 desiredVel = dist * flockp->max_speed;

pt.goal = (desiredVel - velocity_i);
pt.goal.w = 0.f;

#endif
