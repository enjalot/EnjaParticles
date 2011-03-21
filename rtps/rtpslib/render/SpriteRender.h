#ifndef SPRITE_RENDER_H
#define SPRITE_RENDER_H

#include "Render.h"

namespace rtps
{
    class SpriteRender : public Render
    {
    public:
        SpriteRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings);
        ~SpriteRender();
        virtual void render();
    };
};

#endif
