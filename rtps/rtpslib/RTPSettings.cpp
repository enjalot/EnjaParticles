#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    max_particles = 1024*4 * 4 * 4 * 2 * 2; 
   // max_particles = 512;
    dt = .0005f;
}

}
