#ifndef _DENSITY_UPDATE_CL_
#define _DENSITY_UPDATE_CL_


	// TODO: set dmin = searchradius initialy somewhere
	
	if(rlen < dmin){
		pt->density.x = index_j; //TODO: get the index
	}	
	pt->density.y += pt->density.y;
	pt->xflock += pt->pos(index_j); 
//----------------------------------------------------------------------
#endif
