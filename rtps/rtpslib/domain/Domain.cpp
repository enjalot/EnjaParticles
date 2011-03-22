#include <math.h>

#include "Domain.h"

namespace rtps
{

    Domain::Domain(float4 min, float4 max)
    {
        this->bnd_min = min;
        this->bnd_max = max;
    }

    void Domain::calculateCells(float cell_size)
    {
        double s2 = 2.*cell_size;
        min = this->bnd_min - float4(s2, s2, s2, 0.);
        max = this->bnd_max + float4(s2, s2, s2, 0.);

        this->min = min;
        this->max = max;

        printf("cell size: %f\n ASDFASDFSDF\n", cell_size);

        size = float4(max.x - min.x,
                      max.y - min.y,
                      max.z - min.z,
                      0.0f);

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
        /*
        delta = float4(size.x / res.x,
                       size.y / res.y,
                       size.z / res.z,
                       1.0f);
        */

    }

    Domain::~Domain()
    {
    }

}
