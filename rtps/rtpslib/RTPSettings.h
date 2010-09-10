#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED



namespace rtps{

class RTPSettings
{
public:
    RTPSettings();

    //decide which system to use
    enum SysType {Simple, SPH, BOIDS};
    SysType system;

    int max_particles;

};

}

#endif
