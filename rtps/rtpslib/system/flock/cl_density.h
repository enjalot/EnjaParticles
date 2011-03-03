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

	// setup for rule 1. Separation
	float4 s = pj - pi;
	float  d = distance(pi,pj);
	float  r = d / mindist;
	
	s = normalize(s);

	if(d > mindist){ 
		s *= r;
	}
	else if(d < mindist){
		s *= -r;
	}
	else{
		s *= 0.f;
	}

	// force is the separation vector
	pt->force += s;


	// setup for rule 2. alignment
	// surf_tens is the alignment vector
	pt->surf_tens += vj;


	// setup for rule 3. cohesion
	pt->density.x += 1.f;		// number of flockmates

	pt->xflock += pj; 		// center of the flock
	pt->xflock.w = 1.f;
     }
//----------------------------------------------------------------------
#endif
