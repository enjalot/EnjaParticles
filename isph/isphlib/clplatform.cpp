#include "isph.h"
using namespace isph;

CLPlatform::CLPlatform(cl_platform_id platform) :
	id(platform),
	deviceCount(0),
	devices(NULL)
{

}

CLPlatform::~CLPlatform()
{
	for (cl_uint i=0; i<deviceCount; i++)
		delete devices[i];
}
