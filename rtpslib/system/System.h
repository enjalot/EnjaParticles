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


#ifndef RTPS_SYSTEM_H_INCLUDED
#define RTPS_SYSTEM_H_INCLUDED

#include "../domain/Domain.h"
#include "ForceField.h"
#include "../render/Render.h"
#include "../render/SpriteRender.h"
#include "../render/SSFRender.h"
#include "../render/Sphere3DRender.h"

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

#include<stdio.h>
namespace rtps
{

    class RTPS_EXPORT System
    {
    public:
        virtual void update() = 0;

        virtual ~System()
        {
            delete renderer;
        }

        virtual Domain* getGrid()
        {
            return grid;
        }
        virtual int getNum()
        {
            return num;
        }
        virtual void setNum(int nn)
        {
            num = nn;
        };//should this be public
        virtual GLuint getPosVBO()
        {
            return pos_vbo;
        }
        virtual GLuint getColVBO()
        {
            return col_vbo;
        }

        virtual void render()
        {
            renderer->render();
        }

/*
        template <typename RT>
        virtual RT GetSettingAs(std::string key, std::string defaultval = "0") 
        {
        };
        template <typename RT>
        virtual void SetSetting(std::string key, RT value) 
        {
        };
*/

        virtual int addBox(int nn, float4 min, float4 max, bool scaled, float4 color=float4(1., 0., 0., 1.))
        {
            return 0;
        };

        virtual void addBall(int nn, float4 center, float radius, bool scaled, float4 color=float4(1., 0., 0., 1.))
        {
        };
        virtual int addHose(int total_n, float4 center, float4 velocity, float radius, float4 color=float4(1., 0., 0., 1.))
        {
            return 0;
        };
        virtual void updateHose(int index, float4 center, float4 velocity, float radius, float4 color=float4(1., 0., 0., 1.))
        {
        };
        virtual void refillHose(int index, int refill)
        {
        };
 
        /*
        virtual void addHose(int total_n, float4 center, float4 velocity, float radius, float spacing)
        {
        };
        */
        virtual void sprayHoses()
        {
        };
        virtual void testDelete()
        {
        };


        virtual void loadTriangles(std::vector<Triangle> &triangles)
        {
        };
        virtual void addForceField(ForceField ff)
        {
        };


        virtual void printTimers()
        {
            renderer->printTimers();
        };

        virtual Render* getRenderer()
        {
            return renderer;
        }

    protected:
        //number of particles
        int num;  // USED FOR WHAT? 
        //maximum number of particles (for array allocation)
        int max_num;
        //maximum number of outer particles (for array allocation)

        GLuint pos_vbo;
        GLuint col_vbo;
        //flag is true if the system's constructor creates the VBOs for the system
        bool managed;

        Domain* grid;

        Render* renderer;

        std::string resource_path;
        std::string common_source_dir;

        virtual void setRenderer()
        {
            //delete renderer;
            //renderer = render;
        }

    };

}

#endif
