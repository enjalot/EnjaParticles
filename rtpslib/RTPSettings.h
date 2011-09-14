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
    //next largest power of 2. hack required for BitonicSort
    unsigned int nlpo2(register unsigned int x);

    class RTPS_EXPORT RTPSettings 
    {
    public:
        //decide which system to use
        enum SysType
        {
            Simple, SPH, FLOCK, OUTER
        };
        SysType system;

        enum RenderType
        {
            RENDER = 0, SPRITE_RENDER, SCREEN_SPACE_RENDER, SPHERE3D_RENDER
        };


        RTPSettings();
        RTPSettings(SysType system, int max_particles, float dt);
        RTPSettings(SysType system, int max_particles, float dt, Domain *grid);

        //collision
        RTPSettings(SysType system, int max_particles, float dt, Domain *grid, bool tri_collision);

        //flock
        RTPSettings(SysType system, int max_particles, float dt, Domain* grid, float maxspeed, float mindist, float searchradius, float color[], float w_sep, float w_align, float w_coh);

		// (GE) TEMPORARY WHILE WAITING FOR REFACTORING
		void setMaxOuterParticles(int max_outer_particles) {
			this->max_outer_particles = nlpo2(max_outer_particles);
		}
		int getMaxOuterParticles() {
			return this->max_outer_particles;
		}

        //without this, windows was crashing with a ValidHeapPointer
        //assertion error. Indicates the heap may be corrupted by 
        //something in here
        ~RTPSettings();

        //TODO get rid of all variables, just use map
        //maximum number of particles a system can hold
        int max_particles;
        //maximum number of outer (ghost, solid object) particles a system can hold
        int max_outer_particles; 
        //the bounding domain of the system
        Domain *grid; //TODO keep this and make private
        //time step per iteration
        float dt;
        //triangle collision?
        bool tri_collision;

        // FLOCK: target of goal rule
        float4 target;

        // FLOCK: 2D simulation
        bool two_dimensional;

        bool has_changed() { return changed; };
        void updated() { changed = false; }; //for now we are assuming only one consumer (one system using the settings)

        void printSettings();

		//----------------------------------------------------------------------
        // Return the value associate with KEY as the specified template parameter type
        // e.g.,
        //  int i = SPHSettings.GetSettingAs<int>("key");
        //  double d = SPHSettings.GetSettingAs<double>("key2");
        //  string s = SPHSettings.GetSettingAs<string>("key3");
        template <typename RT>
        RT GetSettingAs(std::string key, std::string defaultval = "0") 
        {
            if (settings.find(key) == settings.end()) 
            {
                RT ret = ss_typecast<RT>(defaultval);
                return ret;
            }
            return ss_typecast<RT>(settings[key]);
        }

		//----------------------------------------------------------------------
        template <typename RT>
        void SetSetting(std::string key, RT value) {
            // TODO: change to stringstream for any type of input that is cast as string
            std::ostringstream oss; 
            oss << value; 
            settings[key] = oss.str(); 
            std::cout << "setting: " << key << " | " << value << std::endl;//printf("setting: %s %s\n", settings[key].c_str());
            changed = true;
        }
    
		//----------------------------------------------------------------------
        bool Exists(std::string key) { if(settings.find(key) == settings.end()) { return false; } else { return true; } }

    private:
        std::map<std::string, std::string> settings;
        bool changed;
        
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
        Domain* getDomain()
        {
            return grid;
        }
        void setDomain(Domain *domain)//should this pass by reference?
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

        float4 getTarget()
        {
            return target;
        }
        void setTarget(float4 t)
        {
            target = t;
            target.print("target");
        }

        int getDimension()
        {
            return two_dimensional;
        }
        void setDimension(bool dim)
        {
            two_dimensional = dim;
        }
    };

}

#endif
