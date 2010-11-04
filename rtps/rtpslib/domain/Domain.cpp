#include <math.h>

#include "Domain.h"

namespace rtps {

Domain::Domain(float4 min, float4 max)
{
    this->min = min;
    this->max = max;
    size = float4(max.x - min.x,
                  max.y - min.y,
                  max.z - min.z,
                  0.0f);
}

void Domain::calculateCells(float cell_size)
{
    res = float4(ceil(size.x / cell_size),
                 ceil(size.y / cell_size),
                 ceil(size.z / cell_size),
                 0.0f);

    size = float4(res.x * cell_size,
                  res.y * cell_size,
                  res.z * cell_size,
                  0.0f);

    delta = float4(res.x / size.x,
                   res.y / size.y,
                   res.z / size.z,
                   0.0f);
}

Domain::~Domain()
{
}

}
