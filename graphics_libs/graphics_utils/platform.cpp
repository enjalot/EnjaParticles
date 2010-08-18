//
// platform.cpp
//
// 2004 Patrick Crawley
//

#include "platform.h"
#include <cstdlib>

#ifdef WIN32
	// placeholder for consistancy
#else
	#include <unistd.h>
#endif

void srand_float(int i)
{
	if (i == 0)
		#ifdef WIN32
		i = get_clock_ticks(); /// TODO : this is incorrect; fix when i get to the windows part
		#else
		i = getpid();
		#endif

	srand(i);
}
//----------------------------------------------------------------------
bool isNvidiaCard()
{
#ifdef NVIDIA_VID_CARD
	return true;
#else
	return false;
#endif
}
//----------------------------------------------------------------------
