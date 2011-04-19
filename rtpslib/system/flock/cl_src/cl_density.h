#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_

if(index_i != index_j){
	
	// positions
	float4 pj = pos(index_j);
	
    // velocities
	float4 vj = vel(index_j);

	// number of flockmates
    pt->density.x += 1.f;

	// setup for Rule 1. Separation
	// force is the separation vector
    float4 s = r;       //pi - pj;
	float  d = rlen;    //length(s);
	
    if(smooth_dist >= min_dist && d <= min_dist){
		s.w = 0.0f;
        s = normalize(s);
        s /= d;
	    pt->force += s;        // accumulate the separation vector
        pt->density.y += 1.f;  // count how many flockmates are with in the separation distance
	}

	// setup for rule 2. alignment
	// surf_tens is the alignment vector
    pt->surf_tens  += vj;   // desired velocity
	pt->surf_tens.w = 1.f;

	// setup for rule 3. cohesion
    // xflock is the cohesion vector
    pt->xflock  += pj; 		// center of the flock
	pt->xflock.w = 1.f;
}
#endif
