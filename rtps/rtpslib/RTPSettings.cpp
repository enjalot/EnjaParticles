#include "RTPSettings.h"
namespace rtps {


RTPSettings::RTPSettings()
{
    system = SPH;
	int n1k = 1024 * 1;
	int n2k = 1024 * 2;
	int n4k = 1024 * 4;
	int n8k = 1024 * 8;
	int n16k = 1024 * 16;
	int n32k = 1024 * 32;
	int n64k = 1024 * 64;
	int n128k = 1024 * 128;
	int n256k = 1024 * 256;

    //max_particles = n4k;
    //max_particles = 256;
    max_particles = 512;
    max_particles = n4k;
    //dt = .0004f;
    //dt = .001f;

	// appears to be ok
	dt = .0011f;

	// appears to be ok
	dt = .0001f;
}

}
