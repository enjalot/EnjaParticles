
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

//notice we treat Vec4s like Vec3 when doing operation so we don't mess with values that have been packed in
//like in velocity

float dot(Vec4 a, Vec4 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vec4 normalize(Vec4 a)
{
    float mag = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); //store the magnitude of the velocity
    return Vec4(a.x/mag, a.y/mag, a.z/mag, a.w);
}

Vec4 cross(Vec4 a, Vec4 b)
{
    return Vec4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

Vec4 sub(Vec4 a, Vec4 b)
{
    return Vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

Vec4 scala(float s, Vec4 a)
{
    return Vec4(s*a.x, s*a.y, s*a.z, s*a.w);
}

//Moller and Trumbore
bool intersect_triangle(Vec4 pos, Vec4 vel, Vec4 tri[3], Vec4 triN, float dist)
{
    //take in the particle position and velocity (treated as a Ray)
    //also the triangle vertices for the ray intersection
    //we take in the precalculated triangle's normal to first test for distance
    //dist is the threshold to determine if we are close enough to the triangle
    //to even check for distance
    Vec4 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det, u, v;
    float eps = .000001;

    //check distance
    tvec = sub(pos, tri[0]);
    float distance = -dot(tvec, triN) / dot(vel, triN);
    if (distance > dist)
        return false;


    edge1 = sub(tri[1], tri[0]);
    edge2 = sub(tri[2], tri[0]);

    pvec = cross(vel, edge2);
    det = dot(edge1, pvec);
    //culling branch
    //if(det > -eps && det < eps)
    if(det < eps)
        return false;

    u = dot(tvec, pvec);
    if (u < 0.0 || u > det)//1.0)
        return false;

    qvec = cross(tvec, edge1);
    v = dot(vel, qvec);
    if (v < 0.0 || u + v > det)//1.0f)
        return false;

    return true;
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
    
    //COLLISION STUFF
        //set up test plane
        Vec4 plane[4];
        plane[0] = Vec4(-2,-1,-2,0);
        plane[1] = Vec4(-2,-1,2,0);
        plane[2] = Vec4(2,-2,2,0);
        plane[3] = Vec4(2,-1,-2,0);

        //triangle fan from plane (for handling faces)
        Vec4 tri[3];
        tri[0] = plane[0];
        tri[1] = plane[1];
        tri[2] = plane[2];

        // do 1 triangle first
        //Vec4 tri2[3];
        //tr2[0] = plane[0];
        //tr2[1] = plane[2];
        //tr2[2] = plane[3];
        //
        Vec4 A = tri[0];
        Vec4 B = tri[1];
        Vec4 C = tri[2];
        Vec4 bma = Vec4(B.x - A.x, B.y - A.y, B.z - A.z, 0);
        Vec4 cma = Vec4(C.x - A.x, C.y - A.y, C.z - A.z, 0);
        //triangle normal should come from blender
        Vec4 triN = normalize(cross(bma, cma));
        printf("triN.x: %f triN.y %f triN.z %f\n", triN.x, triN.y, triN.z);
        //Vec4 triN = cross_product(B - A, C - A);
        //Vec4 triN = Vec4(0.0, 1.0, 0.0, 0.0); 


    //COLLISION STUFF



    printf("about to start cpu loop\n");
    float h = dt;
    for(int i = 0; i < num; i++)
    {
        float life = velocities[i].w;
        life -= h/10;    //should probably depend on time somehow
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

        Vec4 pos = vertices[i];
        Vec4 vel = velocities[i];

        float xn = pos.x;
        float yn = pos.y;
        float zn = pos.z;


        float vxn = vel.x;
        float vyn = vel.y;
        float vzn = vel.z;
        vel.x = vxn;
        vel.y = vyn - h*9.8;
        vel.z = vzn;// - h*9.8;

        xn += h*vel.x;
        yn += h*vel.y;
        zn += h*vel.z;
        
        if(intersect_triangle(pos, vel, tri, triN, h))
        {
            //particle and triangle collision
            //if (yn < -1.0f)

            //lets do some specular reflection
            float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
            Vec4 nvel = normalize(vel);
            float s = 2.0f*(dot(triN, nvel));
            Vec4 dir = scala(s, triN);
            dir = sub(dir, nvel);
            float damping = .5f;
            mag *= damping;
            printf("orig vel: %f %f %f\n", vel.x, vel.y, vel.z);
            vel = scala(-mag, dir);
            printf("new vel: %f %f %f\n", vel.x, vel.y, vel.z);

            xn = pos.x + h*vel.x;
            yn = pos.y + h*vel.y;
            zn = pos.z + h*vel.z;



            /*
            //based on nvidia particle example, sphere-sphere collision
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
            
            velA.x += force.x;
            velA.y += force.y;
            velA.z += force.z;
            
            xn = pos.x + h*velA.x;
            yn = pos.y + h*velA.y;
            zn = pos.z + h*velA.z;
            velocities[i] = velA;
            */
        }
        //}

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
        velocities[i] = vel;
        //save the life!
        velocities[i].w = life;
        //((Vec4*)vertices_p)[i] = vertices[i];
    }
    //push to vbos
    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec4)*num, &vertices[0], GL_DYNAMIC_DRAW); // upload data to video card
    glFinish();

    glBegin(GL_TRIANGLES);
    glVertex3f(tri[0].x, tri[0].y, tri[0].z);
    glVertex3f(tri[1].x, tri[1].y, tri[1].z);
    glVertex3f(tri[2].x, tri[2].y, tri[2].z);

    glEnd();

}

