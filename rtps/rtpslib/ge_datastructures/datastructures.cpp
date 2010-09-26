#include "datastructures.h"
#include <string>
using namespace std;

namespace rtps {

//----------------------------------------------------------------------
DataStructures::DataStructures(RTPS*  ps)
{
	this->ps = ps;
}
//----------------------------------------------------------------------

#include "build_datastructures.cpp"
#include "setup_arrays.cpp"
#include "hash.cpp"
#include "sort.cpp"
#include "neighbor_search.cpp"

}
//----------------------------------------------------------------------
