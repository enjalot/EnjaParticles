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


#include <stdio.h>

#include <GL/glew.h>

#include "System.h"
#include "Simple.h"

namespace rtps
{


    Simple::Simple(RTPS *psfr, int n)
    {
        max_num = n;
        num = max_num;
        //store the particle system framework
        ps = psfr;
        grid = ps->settings->grid;
        forcefields_enabled = true;
        max_forcefields = 100;

        printf("num: %d\n", num);
        positions.resize(max_num);
        colors.resize(max_num);
        forces.resize(max_num);
        velocities.resize(max_num);
        //forcefields.resize(max_forcefields);

        float4 min = grid->getBndMin();
        float4 max = grid->getBndMax();

        float spacing = .1; 
        std::vector<float4> box = addRect(num, min, max, spacing, 1);
        std::copy(box.begin(), box.end(), positions.begin());


        float4 center = float4(1,1,.1,1);
        float4 center2 = float4(2,2,.1,1);
        float4 center3 = float4(1,2,.1,1);

        forcefields.push_back( ForceField(center2, .5,.1) );
        forcefields.push_back( ForceField(center, .5, .1) );
        forcefields.push_back( ForceField(center3, .5, .1) );

        //forcefields.push_back( ForceField(center, 1., 20, 0, 0) );
        //forcefields.push_back( ForceField() );
        //forcefields.push_back( ForceField() );


        //std::fill(positions.begin(), positions.end(), float4(0.0f, 0.0f, 0.0f, 1.0f));
        std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.0f));
        std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

        managed = true;
        pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("pos vbo: %d\n", pos_vbo);
        col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("col vbo: %d\n", col_vbo);

#if GPU
        //vbo buffers
        printf("making cl_buffers\n");
        cl_position = Buffer<float4>(ps->cli, pos_vbo);
        cl_color = Buffer<float4>(ps->cli, col_vbo);
        printf("done with cl_buffers\n");
        //pure opencl buffers
        cl_force = Buffer<float4>(ps->cli, forces);
        cl_velocity = Buffer<float4>(ps->cli, velocities);;

        //could generalize this to other integration methods later (leap frog, RK4)
        printf("create euler kernel\n");
        loadEuler();


        printf("load forcefiels");
        loadForceField();   
        loadForceFields(forcefields);




#endif

        renderer = new Render(pos_vbo,col_vbo,n,ps->cli);
    }

    Simple::~Simple()
    {
        if (pos_vbo && managed)
        {
            glBindBuffer(1, pos_vbo);
            glDeleteBuffers(1, (GLuint*)&pos_vbo);
            pos_vbo = 0;
        }
        if (col_vbo && managed)
        {
            glBindBuffer(1, col_vbo);
            glDeleteBuffers(1, (GLuint*)&col_vbo);
            col_vbo = 0;
        }
    }

    void Simple::update()
    {
#ifdef CPU

        cpuForceField();
        //printf("calling cpuEuler\n");
        cpuEuler();

        //printf("pushing positions to gpu\n");
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float4), &colors[0], GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glFinish();

        //printf("done pushing to gpu\n");


#endif
#ifdef GPU

        //call kernels
        //add timings
        glFinish();
        cl_position.acquire();
        cl_color.acquire();

        //k_forcefield.execute(num, 128);
        k_forcefield.execute(num);
        k_euler.execute(num);

        cl_position.release();
        cl_color.release();
#endif
    }


}
