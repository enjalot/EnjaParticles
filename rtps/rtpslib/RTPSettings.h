#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED



namespace rtps{

class RTPSettings
{
public:

    //decide which system to use
    enum SysType {Simple, SPH, BOIDS};
    SysType system;

    RTPSettings();
    RTPSettings(SysType system, int max_particles, float dt);

    //maximum number of particles a system can hold
    int max_particles;

    //time step per iteration
    float dt;

};

}

#endif
