#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    //max_particles = 1024*4 * 4 * 4 * 2 * 2;  // 256k
    //max_particles = 1024*4 * 4 * 4;   // 64k
    //max_particles = 1024*4 * 4;   // 16k
    max_particles = 1024;
    dt = .005f;
}

}
