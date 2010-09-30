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

}
