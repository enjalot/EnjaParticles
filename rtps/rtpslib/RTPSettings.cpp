#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 1024;
    dt = .005f;
}

}
