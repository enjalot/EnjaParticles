
#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #include <GLUT/glut.h>
    #include <OpenGL/CGLCurrent.h> //is this really necessary?
#else
    //OpenGL stuff
    #include <GL/glx.h>
#endif

#include "string.h"
#include "math.h"
#include "../enja.h"

//This is one mess of a file, just trying to get a quick CPU implementation of collision detection
//so I can figure out where my math is wrong
//
//eventually I should clean this up so it can serve as a comparison against OpenCL
//but that means more refactoring of the whole lib I think


float dot(Vec4 a, Vec4 b)
{
}

Vec4 normalize(Vec4 a)
{
}

int EnjaParticles::cpu_update()
{
    printf("in cpu_update\n");
    
    //make a vector of Vec4s to store the current vertices
    AVec4 vertices(num);
    //copy the data from the OpenGL VBO so we can work on it in cpu
    glBindBufferARB(GL_ARRAY_BUFFER, v_vbo);    
    GLvoid* vertices_p = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    memcpy(&vertices[0], vertices_p, sizeof(Vec4)*num);

    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    

    printf("about to start cpu loop\n");
    float h = dt;
    for(int i = 0; i < num; i++)
    {
        float life = velocities[i].w;
        life -= h;    //should probably depend on time somehow
        if(life <= 0.)
        {
            //reset this particle
            vertices[i].x = vert_gen[i].x;
            vertices[i].y = vert_gen[i].y;
            vertices[i].z = vert_gen[i].z;

            velocities[i].x = velo_gen[i].x;
            velocities[i].y = velo_gen[i].y;
            velocities[i].z = velo_gen[i].z;
            life = 1.;
        } 
        float xn = vertices[i].x;
        float yn = vertices[i].y;
        float zn = vertices[i].z;


        float vxn = velocities[i].x;
        float vyn = velocities[i].y;
        float vzn = velocities[i].z;
        velocities[i].x = vxn;
        velocities[i].y = vyn - h*9.8;
        velocities[i].z = vzn;// - h*9.8;

        xn += h*velocities[i].x;
        yn += h*velocities[i].y;
        zn += h*velocities[i].z;
        

        //set up test plane
        Vec4 plane[4];
        plane[0] = Vec4(0,-1,0,0);
        plane[1] = Vec4(0,-1,2,0);
        plane[2] = Vec4(2,-1,2,0);
        plane[3] = Vec4(2,-1,0,0);

        //triangle fan from plane (for handling faces)
        Vec4 tri1[3];
        tri1[0] = plane[0];
        tri1[1] = plane[1];
        tri1[2] = plane[2];

        // do 1 triangle first
        //Vec4 tri2[3];
        //tr2[0] = plane[0];
        //tr2[1] = plane[2];
        //tr2[2] = plane[3];
        //

        //calculate the normal of the triangle
        //might not need to do this if we just have plane's normal
        Vec4 A = tri1[0];
        Vec4 B = tri1[1];
        Vec4 C = tri1[2];
        
        Vec4 pos = vertices[i];
        Vec4 vel = velocities[i];
        //Vec4 tri1N = cross_product(B - A, C - A);
        Vec4 tri1N = Vec4(0.0, 1.0, 0.0, 0.0); 
        //calculate the distnace of pos from triangle
        Vec4 tmp = Vec4(pos.x - A.x, pos.y - A.y, pos.z - A.z, pos.w);
        float distance = -dot(tmp, tri1N) / dot(velocities[i], tri1N);
        Vec4 P = pos + distance*velocities[i];
        if (distance <= 0.0f)
        {
            //particle is past the plane so don't do anything
            printf("i:%d", i);
        }
        else
        {
            int x = 0;
            int y = 2;
            //these should be projections...
            //or at least calculated from the dominant axis of the normal
            //also these are supposed to be float2 but oh well
            Vec4 Ap = Vec4(A.x, A.z, 0, 0);
            Vec4 Bp = Vec4(A.x, A.z, 0, 0);
            Vec4 Cp = Vec4(A.x, A.z, 0, 0);
            Vec4 Pp = Vec4(A.x, A.z, 0, 0);

            Vec4 b = Vec4(Bp.x - Ap.x, Bp.y - Ap.y, 0, 0);
            Vec4 c = Vec4(Cp.x - Ap.x, Cp.y - Ap.y, 0, 0);
            Vec4 p = Vec4(Pp.x - Ap.x, Pp.x - Pp.y, 0, 0);
            
            float u = (p.y*c.x - p.x*c.y)/(b.y*c.x - b.x*c.y);
            float v = (p.y*b.x - p.x*b.y)/(c.y*b.x - c.x*b.y);
            
            if(u >= 0 and v >= 0 and u+v <= 1)
            {

                //particle and triangle collision
                //if (yn < -1.0f)

                Vec4 posA = vertices[i];
                float radiusA = 5.0f;
                Vec4 posB;
                posB.x = vertices[i].x + radiusA/2.0f;
                posB.y = vertices[i].y + radiusA/2.0f;
                posB.z = vertices[i].z + radiusA/2.0f;
                posB.w = vertices[i].w;

                Vec4 relPos = Vec4(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0);
                float dist = sqrt(relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
                float collideDist = radiusA + 0.0;//radiusB;


                //Vec4 norm = (-0.707106, 0.707106, 0.0, 0.0); //this is actually a unit vector
                Vec4 norm = Vec4(0.0, 1.0, 0.0, 0.0); 
                norm = normalize(norm);
                Vec4 velA = velocities[i];    //velocity of particle
                Vec4 velB = Vec4(0,0,0,0);  //velocity of object
                Vec4 relVel = Vec4(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0);

                float relVelDotNorm = relVel.x * norm.x + relVel.y * norm.y + relVel.z * norm.z;
                Vec4 tanVel = Vec4(relVel.x - relVelDotNorm * norm.x, relVel.y - relVelDotNorm * norm.y, relVel.z - relVelDotNorm * norm.z, 0);
                Vec4 force = Vec4(0,0,0,0);
                float springFactor = -.5;//-spring * (collideDist - dist);
                float damping = 1.0f;
                float shear = 1.0f;
                float attraction = 1.0f;
                force = Vec4(
                    springFactor * norm.x + damping * relVel.x + shear * tanVel.x + attraction * relPos.x,
                    springFactor * norm.y + damping * relVel.y + shear * tanVel.y + attraction * relPos.y,
                    springFactor * norm.z + damping * relVel.z + shear * tanVel.z + attraction * relPos.z,
                    0
                );

                //float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
                //vel /= mag;
                //vel = 2.f*(dot(normal, vel))*normal - vel;
                ////vel *= mag; //we know the direction lets go the right speed
                
                velA += force;
                
                xn = vertices[i].x + h*velA.x;
                yn = vertices[i].y + h*velA.y;
                zn = vertices[i].z + h*velA.z;
                velocities[i] = velA;
            }
        }

        vertices[i].x = xn;
        vertices[i].y = yn;
        vertices[i].z = zn;

         
        /*
        colors[i].x = 1.f;
        colors[i].y = life;
        colors[i].z = life;
        //colors[i].w = 1-life;
        colors[i].w = 1;
        */
        //save the life!
        velocities[i].w = life;
        //((Vec4*)vertices_p)[i] = vertices[i];
    }
    //push to vbos
    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec4)*num, &vertices[0], GL_DYNAMIC_DRAW); // upload data to video card
    glFinish();

}

