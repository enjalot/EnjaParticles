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



#include "GL/glew.h"

#include "Sphere3DRender.h"

using namespace std;
namespace rtps
{
    Sphere3DRender::Sphere3DRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {
    	pos_vbo = pos;
    	col_vbo = col;
    	num = n;
		qu = gluNewQuadric();
		if (qu == 0) {
			printf("Insufficient memory for quadric allocation\n");
		}
    }
	//----------------------------------------------------------------------
    void Sphere3DRender::render()
    {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        //printf("BLENDING: %d\n", blending);

        glEnable(GL_LIGHTING);

	//printf("enter Sphere3DRender::render\n");

    #if 1
    //glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    //glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    float* ptr = (float*) glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
	//printf("ptr= %d\n", (long) ptr);

    //printf("Pos PTR[400]: %f\n", ((float*)ptr)[400]);
    //printf("Pos PTR[401]: %f\n", ((float*)ptr)[401]);
    //printf("Pos PTR[402]: %f\n", ((float*)ptr)[402]);
	int count = 0;
	//float* fp = (float*) ptr;
	//printf("num= %d\n", num); exit(0);

	// Need to take blender scale into account

    //glEnableClientState(GL_COLOR_ARRAY);
	for (int i=0; i < num; i++, count+=4) {
    	//printf("Pos PTR: %f, %f, %f\n", ((float*)ptr)[i], ((float*)ptr)[i+1], ((float*)ptr)[i+2]);
		glPushMatrix();
		glTranslatef(ptr[count], ptr[count+1], ptr[count+2]);
		//float dens = ptr[count+3];
		//float scale = 1.5*pow(dens, -1./3.);  // could be done on GPU
		float scale = .06;
		scale *= 1.5;
		//printf("dens= %f, scale= %f\n", dens, scale);
		glScalef(scale, scale, scale);
		glColor3f(.0, 1.0, .0);

        float radius_scale = settings->getRadiusScale(); //GE
		//printf("radius_scale= %f\n", radius_scale);
		//radius_scale = 10.;
		gluSphere(qu, radius_scale, 10, 10); // radius, slices, stacks
		glPopMatrix();
	}
    //glDisableClientState(GL_COLOR_ARRAY);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    #endif

	//printf("after unmapBufferARB\n");
    
    //printf("enable client state\n");

		#if 0
        if (blending)
        {
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            string afunc = settings->GetSettingAs<string>("render_alpha_function");
            if(afunc == "alpha")
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            else if(afunc == "add")
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            else if(afunc == "multiply")
            {
                glBlendFunc(GL_DST_COLOR, GL_ZERO);
                //glBlendFunc(GL_ZERO, GL_SRC_COLOR);
                //glBlendColor(.9, .9, .9, 1.);
                //glBlendFunc(GL_CONSTANT_COLOR, GL_ZERO);
                //glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_ONE);
            }
        }
		#endif

        glDisable(GL_LIGHTING);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);


        //glBindTexture(GL_TEXTURE_2D,gl_textures["texture"]);
        //renderPointsAsSpheres();
        //glBindTexture(GL_TEXTURE_2D,0);


        //glDepthMask(GL_TRUE);

        //glDisable(GL_POINT_SMOOTH);
        if (blending)
        {
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
 
        glPopClientAttrib();
        glPopAttrib();
        
        glFinish();
    }
}
