/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include "RTPSettings.h"
namespace rtps
{
    unsigned int nlpo2(register unsigned int x)
    {
        x--;
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(x+1);
    }

    RTPSettings::RTPSettings()
    {
		printf("rtpsettings: 1\n");
        changed = false;
        system = SPH;
        max_particles = 2048;
        max_outer_particles = 0;
        dt = .001f;
        grid = new Domain(float4(-5,-.3f,0,0), float4(2, 2, 12, 0));
    }

    RTPSettings::RTPSettings(SysType system, int max_particles, float dt)
    {
		printf("rtpsettings: 2\n");
        changed = false;
        this->system = system;
        this->max_particles = max_particles;
        this->max_outer_particles = 0;
        this->dt = dt;
        grid = new Domain(float4(-5,-.3f,0,0), float4(2, 2, 12, 0));
    }

    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid)
    {
		printf("rtpsettings: 3\n");
        changed = false;
        this->system = system;
        this->max_particles = nlpo2(max_particles);
        this->max_outer_particles = 0;
        this->dt = dt;
        this->grid = grid;
    }

    //with triangle collision
    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid, bool tri_collision)
    {
		printf("rtpsettings: 4\n");
        changed = false;
        this->system = system;
        this->max_particles = max_particles;
        this->max_outer_particles = 0;
        this->dt = dt;
        this->grid = grid;
        this->tri_collision = tri_collision;
    }
    
#if 0
    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid, float maxspeed, float mindist, float searchradius, float color[], float w_sep, float w_align, float w_coh)
    {
		printf("rtpsettings: 5\n");
        changed = false;
        this->system = system;
        this->system = system;
        this->max_particles = max_particles;
        this->max_outer_particles = 0;
        this->dt = dt;
        this->grid = grid;
        this->max_speed = maxspeed;
        this->min_dist = mindist;
        this->search_radius = searchradius;
        this->color = float4(color[0]/255, color[1]/255, color[2]/255,1.f);
        this->w_sep = w_sep;
        this->w_align = w_align;
        this->w_coh = w_coh;
    }
#endif

    RTPSettings::~RTPSettings()
    {
        printf("settings destructing!\n");
    }

    void RTPSettings::printSettings()
    {
        printf("RTPS Settings\n");
        typedef std::map <std::string, std::string> MapType;

        MapType::const_iterator end = settings.end();
        for(MapType::const_iterator it = settings.begin(); it != end; ++it)
        {
            printf("%s: %s\n", it->first.c_str(), it->second.c_str());
        }
    }

}
