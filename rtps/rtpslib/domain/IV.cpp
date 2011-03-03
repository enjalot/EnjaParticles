#include "IV.h"
#include <vector>
#include<stdlib.h>
#include<time.h>

namespace rtps
{

std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
{
/*!
 * Create a rectangle with at most num particles in it.
 *  The size of the return vector will be the actual number of particles used to fill the rectangle
 */
    spacing *= 1.1f;

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
    spacing *= 1.1f;
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

std::vector<float4> addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax)
{
/*!
 * Create a rectangle with at most num particles in it.
 *  The size of the return vector will be the actual number of particles used to fill the rectangle
 */

    srand(time(NULL));	

    spacing *= rand()/RAND_MAX;

    float xmin = min.x * dmin.x; // / scale;
    float xmax = max.x * dmax.x; // / scale;
    float ymin = min.y * dmin.y; // / scale;
    float ymax = max.y * dmax.y; // / scale;
    float zmin = min.z * dmin.z; // / scale;
    float zmax = max.z * dmax.z; // / scale;

    std::vector<float4> rvec(num);
    int i=0;
    for (float z = zmin; z <= zmax; z+=spacing) {
    for (float y = ymin; y <= ymax; y+=spacing) {
    for (float x = xmin; x <= xmax; x+=spacing) {
        if (i >= num) break;				
        rvec[i] = float4(x-rand()/RAND_MAX,y-rand()/RAND_MAX,z-rand()/RAND_MAX,1.0f);
        i++;
    }}}
    rvec.resize(i);
    return rvec;

}


std::vector<float4> addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax)
{
/*!
 * Create a rectangle with at most num particles in it.
 *  The size of the return vector will be the actual number of particles used to fill the rectangle
 */
    srand(time(NULL));	
    
    spacing *= rand()/RAND_MAX;

    float xmin = (center.x - radius) * dmin.x; // / scale;
    float xmax = (center.x + radius) * dmax.x; // / scale;
    float ymin = (center.y - radius) * dmin.y; // / scale;
    float ymax = (center.y + radius) * dmax.y; // / scale;
    float zmin = (center.z - radius) * dmin.z; // / scale;
    float zmax = (center.z + radius) * dmax.z; // / scale;
    
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
        rvec[i] = float4(x-rand()/RAND_MAX,y-rand()/RAND_MAX,z-rand()/RAND_MAX,1.0f);
        i++;
    }}}
    rvec.resize(i);
    return rvec;
}

}
