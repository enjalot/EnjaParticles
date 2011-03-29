#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED

#include <stdlib.h>
#include <string>
#include <map>
#include <iostream>
#include <stdio.h>
#include <sstream>


#include "domain/Domain.h"
#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{

    class RTPS_EXPORT RTPSettings 
    {
    public:
        //decide which system to use
        enum SysType
        {
            Simple, SPH, FLOCK 
        };
        SysType system;

        enum RenderType
        {
            RENDER = 0, SPRITE_RENDER, SCREEN_SPACE_RENDER
        };


        RTPSettings();
        RTPSettings(SysType system, int max_particles, float dt);
        RTPSettings(SysType system, int max_particles, float dt, Domain grid);

        //collision
        RTPSettings(SysType system, int max_particles, float dt, Domain grid, bool tri_collision);

        //flock
        RTPSettings(SysType system, int max_particles, float dt, Domain grid, float maxspeed, float mindist, float searchradius, float color[]);

        //TODO get rid of all variables, just use map
        //maximum number of particles a system can hold
        int max_particles;
        //the bounding domain of the system
        Domain grid; //TODO keep this and make private
        //time step per iteration
        float dt;
        //triangle collision?
        bool tri_collision;


        // max speed of the boids
        float max_speed;
        // desired separation distance of the boids
        float min_dist;
        // radius to search for flockmates
        float search_radius;
        // color of the flock
        float4 color;


        //bool has_changed();
        void printSettings();

        // Return the value associate with KEY as the specified template parameter type
        // e.g.,
        //  int i = SPHSettings.GetSettingAs<int>("key");
        //  double d = SPHSettings.GetSettingAs<double>("key2");
        //  string s = SPHSettings.GetSettingAs<string>("key3");
        template <typename RT>
        RT GetSettingAs(std::string key, std::string defaultval = "0") 
        {
            //std::cout << "[Project Settings] \t"; 
            if (settings.find(key) == settings.end()) 
            {
                RT ret = ss_typecast<RT>(defaultval);
                //std::cout << key << " = " << ret;
                //std::cout << "\t<-- default value." << std::endl;
                return ret;
            }
            //std::cout << key << " = " << settings[key] << std::endl;
            return ss_typecast<RT>(settings[key]);
        }

        template <typename RT>
        void SetSetting(std::string key, RT value) {
            // TODO: change to stringstream for any type of input that is cast as string
            RT oldval = this->GetSettingAs<RT>(key);
            if (oldval != value)
            {
                std::ostringstream oss; 
                oss << value; 
                settings[key] = oss.str(); 
                //changed = true;
            }
        }

    private:
        std::map<std::string, std::string> settings;
        
        
        // Added by GE
        float render_radius_scale;
        float render_blur_scale;
        RenderType render_type;
        bool use_glsl;
        bool use_alpha_blending;


        // This routine is adapted from post on GameDev:
        // http://www.gamedev.net/community/forums/topic.asp?topic_id=190991
        // Should be safer to use this than atoi. Performs worse, but our
        // hotspot is not this part of the code.
        template<typename RT, typename _CharT, typename _Traits , typename _Alloc >
        RT ss_typecast( const std::basic_string< _CharT, _Traits, _Alloc >& the_string )
        {
            std::basic_istringstream< _CharT, _Traits, _Alloc > temp_ss(the_string);
            RT num;
            temp_ss >> num;
            return num;
        }



    public:
        Domain getDomain()
        {
            return grid;
        }
        void setDomain(Domain domain)//should this pass by reference?
        {
            grid = domain;
        }


        //TODO: remove these when we switch to Map
        float getRadiusScale()
        {
            return render_radius_scale;
        }
        void setRadiusScale(float scale)
        {
            render_radius_scale = scale;
        }

        float getBlurScale()
        {
            return render_blur_scale;
        }
        void setBlurScale(float scale)
        {
            render_blur_scale = scale;
        }

        int getRenderType()
        {
            return render_type;
        }
        void setRenderType(RenderType type)
        {
            render_type = type;
        }

        int getUseAlphaBlending()
        {
            return use_alpha_blending;
        }
        void setUseAlphaBlending(bool use_alpha)
        {
            use_alpha_blending = use_alpha;
        }

        int getUseGLSL()
        {
            return use_glsl;
        }
        void setUseGLSL(bool use_glsl)
        {
            this->use_glsl = use_glsl;
        }
    };

}

#endif
