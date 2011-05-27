#ifndef SPHERE3D_RENDER_H
#define SPHERE3D_RENDER_H

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
    class RTPS_EXPORT Sphere3DRender : public Render
    {
	private:
		GLUquadric *qu;

    public:
        Sphere3DRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings);
        ~Sphere3DRender();
        virtual void render();
    };
};

#endif
