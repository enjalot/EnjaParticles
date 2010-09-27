#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 1024*2*2 * 4;
    dt = .0005f;
}

}
