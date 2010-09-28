#ifndef _TIMERS_H_
#define _TIMERS_H_

#include "timege.h";

namespace rtps {

class Timers
{
public:
	enum {TI_HASH=0, TI_SORT, TI_BUILD, TI_NEIGH, TI_DENS, TI_PRES, TI_EULER};
	GE::Time* ts_cl[10];
};

}
#endif
