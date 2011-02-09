#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_


	// TODO: set dmin = searchradius initialy somewhere
	
	if(rlen < dmin){
		pt->density.x = index_j; // nearest flockmate
	}	
	pt->density.y += 1;		// number of flockmates
	pt->xflock += position_j; 	// center of the flock
//----------------------------------------------------------------------
#endif
