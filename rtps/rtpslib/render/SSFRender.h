#ifndef SSF_RENDER_H
#define SSF_RENDER_H

#include "Render.h"

namespace rtps
{
    class SSFRender : public Render
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
