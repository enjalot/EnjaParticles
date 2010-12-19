#include <stdio.h>

#include <android/log.h>
#include "importgl.h"

#include "System.h"
#include "Simple.h"

namespace rtps {


Simple::Simple(RTPS *psfr, int n)
{
    max_num = n;
    num = max_num;
    //store the particle system framework
    ps = psfr;
    grid = ps->settings.grid;

    /*
    std::vector<float4> positions(num);
    std::vector<float4> colors(num);
    std::vector<float4> forces(num);
    std::vector<float4> velocities(num);
    */
    //printf("num: %d\n", num);
    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);

    int w = 20;
    int j = 0;
    float wf = w*1.0f;
    /*
    for(int i = 0; i < num; i++)
    {
        positions[i] = float4((i % w)/wf, j/wf, 0.0f, 0.0f);
        colors[i] = float4(1.0f, 0.0f, 0.0f, 0.0f);
        if(i % w == 0)
        {
            j++;
        }
    }
    */

    float4 min = grid.getBndMin();
    float4 max = grid.getBndMax();

    __android_log_print(ANDROID_LOG_INFO, "RTPS", "simple constructor in x, y, z: %f, %f, %f", min.x, min.y, min.z);
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "simple constructor max x, y, z: %f, %f, %f", max.x, max.y, max.z);


    float spacing = .1;
    std::vector<float4> box = addRect(num, min, max, spacing, 1);
    std::copy(box.begin(), box.end(), positions.begin());
    //std::fill(positions.begin(), positions.end(), float4(0.0f, 0.0f, 0.0f, 1.0f));
    std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.5f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "addRect p1 x, y, z: %f, %f, %f", positions[1].x, positions[1].y, positions[1].z);
    
    managed = true;
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);

}

Simple::~Simple()
{
    if(pos_vbo && managed)
    {
        glBindBuffer(1, pos_vbo);
        glDeleteBuffers(1, (GLuint*)&pos_vbo);
        pos_vbo = 0;
    }
    if(col_vbo && managed)
    {
        glBindBuffer(1, col_vbo);
        glDeleteBuffers(1, (GLuint*)&col_vbo);
        col_vbo = 0;
    }
}

void Simple::update()
{

    //printf("calling cpuEuler\n");
    cpuEuler();

    //printf("pushing positions to gpu\n");
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_num * sizeof(float4), &colors[0], GL_DYNAMIC_DRAW);

    //printf("done pushing to gpu\n");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

}



}
