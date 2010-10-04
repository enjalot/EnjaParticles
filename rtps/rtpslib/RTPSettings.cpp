#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
    //max_particles = 1024*4 * 4 * 4 * 2 * 2;  // 256k
    //max_particles = 1024*4 * 4 * 4;   // 64k
    //max_particles = 1024*4 * 4;   // 16k
    max_particles = 1024 * 1;
	// Code works only for 1024 particles? 
    //max_particles = 1024;
    dt = .005f;
    dt = .05f;
}

}
