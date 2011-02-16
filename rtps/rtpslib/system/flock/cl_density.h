#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_


	// TODO: set dmin = searchradius initialy somewhere

    if(index_i != index_j){
	
	if(rlen < mindist){
		pt->density.x = index_j; // nearest flockmate
		mindist = rlen;
	}	

	pt->density.y += 1;		// number of flockmates

	pt->xflock += pos(index_j); 	// center of the flock
	pt->xflock.w = 1.f;
     }
//----------------------------------------------------------------------
#endif
