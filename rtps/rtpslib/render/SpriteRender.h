#ifndef SPRITE_RENDER_H
#define SPRITE_RENDER_H

#include "Render.h"

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
    class RTPS_EXPORT SpriteRender : public Render
    {
    public:
        SpriteRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings);
        ~SpriteRender();
        virtual void render();
    };
};

#endif
