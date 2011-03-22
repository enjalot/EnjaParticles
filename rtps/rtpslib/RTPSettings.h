#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED

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
            Simple, SPH, SimpleFlock
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

        //maximum number of particles a system can hold
        int max_particles;
        
        //the bounding domain of the system
        Domain grid;

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

        // Added by GE
    private:
        float render_radius_scale;
        float render_blur_scale;
        RenderType render_type;
        bool use_glsl;
        bool use_alpha_blending;

    public:
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
