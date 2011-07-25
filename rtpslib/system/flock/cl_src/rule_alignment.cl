#ifndef _RULE_ALIGNMENT_CL_
#define _RULE_ALIGNMENT_CL_

// velocities
float4 vj = vel[index_j];
	        
// rule 2. alignment
pt->alignment += vj;   

#endif
