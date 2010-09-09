#ifndef RTPS_RENDER_H_INCLUDED
#define RTPS_RENDER_H_INCLUDED



namespace rtps{

class Render
{
public:
    Render();

    //decide which kind of rendering to use
    enum RenderType {POINTS, SPRITES};
    RenderType rtype;

    void render();
    void drawArrays();

    //void compileShaders();

};

}

#endif
