#ifndef _RULE_SEPARATION_CL_
#define _RULE_SEPARATION_CL_

// rule 1. separation
if(rlen <= flockp->min_dist){ 
    r /= rlen;
    pt->separation += r;        
}

#endif
