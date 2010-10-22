
// Make these routines part of GE_SPH. 
// Ultimately, they should either be a part of the superclass, System, 
// or in some utility class

//#include "datastructures.h"
#include "GE_SPH.h"

#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
int DataStructures::getNbVars()
{
	return nb_vars;
}
//----------------------------------------------------------------------
DataStructures::DataStructures(RTPS*  ps)
{
	this->ps = ps;
}
//----------------------------------------------------------------------

#include "build_datastructures_wrap.cpp"
//#include "setup_arrays.cpp"
#include "hash_wrap.cpp"
#include "sort_wrap.cpp"
#include "neighbor_search_wrap.cpp"

}
//----------------------------------------------------------------------
