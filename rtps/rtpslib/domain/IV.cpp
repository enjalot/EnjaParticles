#include "IV.h"
#include <vector>

namespace rtps
{

std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
{
/*!
 * Create a rectangle with at most num particles in it.
 *  The size of the return vector will be the actual number of particles used to fill the rectangle
 */
    float xmin = min.x / scale;
    float xmax = max.x / scale;
    float ymin = min.y / scale;
    float ymax = max.y / scale;
    float zmin = min.z / scale;
    float zmax = max.z / scale;

    std::vector<float4> rvec(num);
    int i=0;
    for (float z = zmin; z <= zmax; z+=spacing) {
    for (float y = ymin; y <= ymax; y+=spacing) {
    for (float x = xmin; x <= xmax; x+=spacing) {
        if (i >= num) break;				
        rvec[i] = float4(x,y,z,1.0f);
        i++;
    }}}
    rvec.resize(i);
    return rvec;

}


std::vector<float4> addSphere(int num, float4 center, float radius, float spacing, float scale)
{
/*!
 * Create a rectangle with at most num particles in it.
 *  The size of the return vector will be the actual number of particles used to fill the rectangle
 */
    float xmin = (center.x - radius) / scale;
    float xmax = (center.x + radius) / scale;
    float ymin = (center.y - radius) / scale;
    float ymax = (center.y + radius) / scale;
    float zmin = (center.z - radius) / scale;
    float zmax = (center.z + radius) / scale;
    float r2 = radius*radius;
    float d2 = 0.0f;

    std::vector<float4> rvec(num);
    int i=0;
    for (float z = zmin; z <= zmax; z+=spacing) {
    for (float y = ymin; y <= ymax; y+=spacing) {
    for (float x = xmin; x <= xmax; x+=spacing) {
        if (i >= num) break;				
        d2 = (x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) + (z - center.z)*(z - center.z);
        if(d2 > r2) continue;
        rvec[i] = float4(x,y,z,1.0f);
        i++;
    }}}
    rvec.resize(i);
    return rvec;


}

}
