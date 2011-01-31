#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 2048;
    dt = .001f;
    grid = Domain(float4(-5,-.3,0,0), float4(2, 2, 12, 0));
}

RTPSettings::RTPSettings(SysType system, int max_particles)
{
    this->system = system;
    this->max_particles = max_particles;
}

RTPSettings::RTPSettings(SysType system, int max_particles, float dt)
{
    this->system = system;
    this->max_particles = max_particles;
    this->dt = dt;
    grid = Domain(float4(-5,-.3,0,0), float4(2, 2, 12, 0));
}

RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain grid)
{
    this->system = system;
    this->max_particles = max_particles;
    this->dt = dt;
    this->grid = grid;
}

RTPSettings::RTPSettings(int max_particles, float maxspeed, float separationdist, float searchradius, float color_b[])
{
    this->system = Swarm;
    this->max_particles = max_particles;
    this->maxspeed = maxspeed;
    this->separationdist = separationdist;
    this->searchradius = searchradius;

    if((color_b[0] || color_b[1] || color_b[2]) > 1.0f){
        color_b[0] /= 255;
        color_b[1] /= 255;
        color_b[2] /= 255;
    }

    this->color.x = color_b[0];
    this->color.y = color_b[1];
    this->color.z = color_b[2];
    this->color.w = 0.f;

}

}
