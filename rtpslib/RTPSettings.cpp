#include "RTPSettings.h"
namespace rtps
{


    RTPSettings::RTPSettings()
    {
        changed = false;
        system = SPH;
        max_particles = 2048;
        dt = .001f;
        grid = new Domain(float4(-5,-.3f,0,0), float4(2, 2, 12, 0));
    }

    RTPSettings::RTPSettings(SysType system, int max_particles, float dt)
    {
        changed = false;
        this->system = system;
        this->max_particles = max_particles;
        this->dt = dt;
        grid = new Domain(float4(-5,-.3f,0,0), float4(2, 2, 12, 0));
    }

    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid)
    {
        changed = false;
        this->system = system;
        this->max_particles = max_particles;
        this->dt = dt;
        this->grid = grid;
    }

    //with triangle collision
    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid, bool tri_collision)
    {
        changed = false;
        this->system = system;
        this->max_particles = max_particles;
        this->dt = dt;
        this->grid = grid;
        this->tri_collision = tri_collision;
    }
    
    RTPSettings::RTPSettings(SysType system, int max_particles, float dt, Domain* grid, float maxspeed, float mindist, float searchradius, float color[], float w_sep, float w_align, float w_coh)
    {
        changed = false;
        this->system = system;
        this->system = system;
        this->max_particles = max_particles;
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
