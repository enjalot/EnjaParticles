#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED

#include "domain/Domain.h"

namespace rtps{

class RTPSettings
{
public:
    //decide which system to use
    enum SysType {Simple, SPH, Swarm};
    SysType system;

    RTPSettings();
    RTPSettings(SysType system, int max_particles);
    RTPSettings(SysType system, int max_particles, float dt);
    RTPSettings(SysType system, int max_particles, float dt, Domain grid);
    RTPSettings(int max_particles, float maxspeed, float separationdist, float searchradius, float color[]);

    //maximum number of particles a system can hold
    int max_particles;
    //the bounding domain of the system
    Domain grid;

    //time step per iteration
    float dt;

    // boids attributes
    float maxspeed;
    float separationdist;
    float searchradius;
    float4 color;



};

}

#endif
