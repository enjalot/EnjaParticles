#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 1024;
    dt = .001f;
    grid = Domain(float4(0,0,0,0), float4(2, 2, 2, 0));

}

RTPSettings::RTPSettings(SysType system, int max_particles, float dt)
{
    this->system = system;
    this->max_particles = max_particles;
    this->dt = dt;
    grid = Domain(float4(0,0,0,0), float4(2, 2, 2, 0));
}

RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain grid)
{   
    this->system = system;
    this->max_particles = max_particles;
    this->dt = dt;
    this->grid = grid;
}   
    
RTPSettings::RTPSettings(int max_particles, float maxspeed, float separationdist, float perceptionrange, float color[])
{   
}   


}
