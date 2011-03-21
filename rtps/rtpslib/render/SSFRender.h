#ifndef SSF_RENDER_H
#define SSF_RENDER_H

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
    class RTPS_EXPORT SSFRender : public Render
    {
    public:
        SSFRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings);
        ~SSFRender();
        void smoothDepth();
        virtual void render();
        virtual void setWindowDimensions(GLuint width,GLuint height);
    protected:
        virtual void deleteFramebufferTextures();
        virtual void createFramebufferTextures();
    };
};

#endif
