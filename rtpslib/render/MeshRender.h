#ifndef MESH_RENDER_H
#define MESH_RENDER_H

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
    class RTPS_EXPORT MeshRender : public Render
    {
    public:
        MeshRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings);
        ~MeshRender();
        virtual void render();
    };
};

#endif
