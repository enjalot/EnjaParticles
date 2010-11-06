#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 1024;
    dt = .005f;
}

RTPSettings::RTPSettings(SysType system, int max_particles, float dt)
{
    this->system = system;
    this->max_particles = max_particles;
    this->dt = dt;
}

RTPSettings::RTPSettings(int max_particles, float maxspeed, float separationdist, float perceptionrange, float color[])
{
}

}
