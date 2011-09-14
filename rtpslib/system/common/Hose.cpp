/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include "Hose.h"



namespace rtps
{

Hose::Hose(RTPS *ps, int total_n, float4 center, float4 velocity, float radius, float spacing, float4 color)
{
    printf("Constructor!\n");
    this->ps = ps;
    this->total_n = total_n;
    this->center = center;
    this->velocity = velocity;
    this->radius = radius;
    this->spacing = spacing;
    this->color = color;
    em_count = 0;
    n_count = total_n;
    calc_vectors();
    center.print("center");
    velocity.print("velocity");
}

void Hose::update(float4 center, float4 velocity, float radius, float spacing, float4 color)
{
    this->center = center;
    this->velocity = velocity;
    this->radius = radius;
    this->spacing = spacing;
    this->color = color;
    calc_vectors();
}

void Hose::refill(int num)
{
    total_n = num;
    n_count = num;
}

void Hose::calc_vectors()
{
    /*
    v = Vec([1., -u.x/u.y, 0.])
    b = 1.
    a = b*u.x/u.y
    c = -b*(u.x**2 + u.y**2)/(u.y*u.z)
    w = Vec([a, b, c])
    */

    //printf("IN CALC VECTORS\n");
    //printf("velocity %f %f %f %f\n", velocity.x, velocity.y, velocity.z, velocity.w);
    //Need to deal with divide by zero if velocity.y or velocity.z is 0
    //can do this properly by switching things around
    //for now we do my new trusty hack ;)
    if (velocity.y == 0.) velocity.y = .0000001;
    if (velocity.z == 0.) velocity.z = .0000001;
    u = float4(1., -(velocity.x/velocity.y), 0., 1.);
    float b = 1.;
    float a = b*velocity.x/velocity.y;
    float c = -b*(velocity.x*velocity.x + velocity.y*velocity.y)/(velocity.y*velocity.z);
    w = float4(a, b, c, 1.);
    u = normalize(u);
    w = normalize(w);
    //printf("u %f %f %f %f\n", u.x, u.y, u.z, u.w);
    //printf("w %f %f %f %f\n", w.x, w.y, w.z, w.w);
}

void Hose::calc_em()
{
/*
 * rate = dt*mag(v) < spacing ?
 * emit every [spacing / (dt*v)] calls 
 * em = (int) (1 + spacing/dt/mag(v))
 * count every call to spray, emit when count == em, restart counter
 */
    //em = 0;
    //em_count = 0;
    float dt = ps->settings->dt;
    float magv = magnitude(velocity);
    //printf("magv: %f\n", magv);
    //em = (int) (1 + spacing/dt/magv/8.);
    //why do i have to divide by 4?
    em = (int) (1 + spacing/dt/magv/10);
    //printf("em: %d\n", em);
}

std::vector<float4> Hose::spray()
{
    //printf("SPRAY!\n");
    em_count++;
    calc_em();
    //printf("em_count %d em %d n_count %d\n", em_count, em, n_count);
    std::vector<float4> particles;
    if(em_count >= em && n_count > 0)
    {
        //std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);
        //particles = addDisc(n_count, center, u, w, radius, spacing);
        float4 v = velocity * ps->settings->dt;
        particles = addDiscRandom(n_count, center, v, u, w, radius, spacing);
        n_count -= particles.size();
        em_count = 0;
    }
    return particles;
}

}
