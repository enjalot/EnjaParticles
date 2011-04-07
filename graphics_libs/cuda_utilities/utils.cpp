#include <stdlib.h>
#include "utils.h"

float Utils::randFloat(float low, float high)
{
    float t = (float)rand() / (float) RAND_MAX;
    return (1.0f - t) * low + t * high;
}
//----------------------------------------------------------------------

