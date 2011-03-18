#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_


	// TODO: set dmin = searchradius initialy somewhere

    if(index_i != index_j){
	
//	if(rlen < mindist){
//		pt->density.x = index_j; // nearest flockmate
//		mindist = rlen;
//	}	

	// positions
	float4 pi = pos(index_i);
	float4 pj = pos(index_j);

	// velocities
	float4 vj = vel(index_j);

	// number of flockmates
    pt->density.x += 1.f;

	// setup for rule 1. Separation
	float4 s = pi - pj;
	float  d = length(s);
	//float  r = d / mindist;
	
	//s = normalize(s);

	//if(d > mindist){ 
	//	s *= r;
	//}
	if(d < mindist){
		//s *= -r;
        s = normalize(s);
        s /= d;
	    pt->force += s;         // accumulate the separation vector
        pt->density.y += 1.f;   // count how many flockmates are with in the separation distance
	}
	//else{
	//	s *= 0.f;
	//}

	// force is the separation vector
    //pt->force += s;         


	// setup for rule 2. alignment
	// surf_tens is the alignment vector
	pt->surf_tens  += vj;
	pt->surf_tens.w = 1.f;


	// setup for rule 3. cohesion
	pt->xflock  += pj; 		// center of the flock
	pt->xflock.w = 1.f;
     }
//----------------------------------------------------------------------
#endif
