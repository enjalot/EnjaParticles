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

        printf("cell size: %f\n ASDFASDFSDF\n", cell_size);

        //width of grid in each dimension
        size = float4(max.x - min.x,
                      max.y - min.y,
                      max.z - min.z,
                      0.0f);

        //number of cells in each dimension
        res = float4(ceil(size.x / cell_size),
                     ceil(size.y / cell_size),
                     ceil(size.z / cell_size),
                     0.0f);

        //width adjusted for whole number of cells
        size = float4(res.x * cell_size,
                      res.y * cell_size,
                      res.z * cell_size,
                      0.0f);

        //width of cell based on adjusted size
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

        this->min = min;
        this->max = min + size;
        //this->max = max;


    }

    Domain::~Domain()
    {
    }

}
