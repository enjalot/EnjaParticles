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


#include "../SPH.h"

namespace rtps {

    CollisionTriangle::CollisionTriangle(std::string path, CL* cli_, EB::Timer* timer_, int max_triangles)
    {
        cli = cli_;
        timer = timer_;
        std::vector<Triangle> maxtri(max_triangles);
        cl_triangles = Buffer<Triangle>(cli, maxtri);


        triangles_loaded = false;
        printf("create collision triangle kernel\n");
        path += "/collision_tri.cl";
        k_collision_tri = Kernel(cli, path, "collision_triangle");
    } 

    //TODO: avoid need for this function?
    void SPH::loadTriangles(std::vector<Triangle> &triangles)
    {
        collision_tri.loadTriangles(triangles);
    }

    void CollisionTriangle::loadTriangles(std::vector<Triangle> &triangles)
    {
        int n_triangles = triangles.size();
        //printf("n triangles: %d\n", n_triangles);
        //load triangles into cl buffer
        //Triangle is a struct that ends up being 4 float4s
        //cl_triangles = Buffer<Triangle>(ps->cli, triangles);
        cl_triangles.copyToDevice(triangles);
        //printf("Triangle z %f\n", triangles[0].verts[0].z);

        k_collision_tri.setArg(3, cl_triangles.getDevicePtr());     //triangles
        k_collision_tri.setArg(4, n_triangles);                     //number of triangles

        //printf("sizeof(Triangle) = %d\n", (int) sizeof(Triangle));


        //TODO: get local mem size from opencl
        size_t max_loc_memory = 1024 << 4;  // 16k bytes local memory on mac
        int max_tri = max_loc_memory / sizeof(Triangle);
        //max_tri = n_triangles;
        max_tri = 220; // fits in cache
        //printf("max_tri= %d\n", max_tri);

        size_t sz = max_tri*sizeof(Triangle);
        //printf("sz= %zd bytes\n", sz);

        k_collision_tri.setArgShared(7, sz);


        triangles_loaded = true;
        //exit(0);

    }

    void CollisionTriangle::execute(int num,
                                    float dt,
                                    //input
                                    //Buffer<float4>& svars, 
                                    Buffer<float4>& pos_s, 
                                    Buffer<float4>& vel_s, 
                                    Buffer<float4>& force_s, 
                                    //output
                                    //params
                                    Buffer<SPHParams>& sphp,
                                    //debug
                                    Buffer<float4>& clf_debug,
                                    Buffer<int4>& cli_debug)
    {
    
    int local_size = 32;
    //printf("triangles loaded? %d\n", triangles_loaded);
    if(triangles_loaded)
    {
        
        //k_collision_tri.setArg(0, svars.getDevicePtr());
        k_collision_tri.setArg(0, pos_s.getDevicePtr());
        k_collision_tri.setArg(1, vel_s.getDevicePtr());
        k_collision_tri.setArg(2, force_s.getDevicePtr());
        // 1 = triangles
        // 2 = n_triangles
        k_collision_tri.setArg(5, dt);
        k_collision_tri.setArg(6, sphp.getDevicePtr());
        // 5 = local triangles
        // ONLY IF DEBUGGING
        k_collision_tri.setArg(8, clf_debug.getDevicePtr());
        k_collision_tri.setArg(9, cli_debug.getDevicePtr());


        //printf("execute!\n");
        float gputime = k_collision_tri.execute(num, local_size);
        if(gputime > 0)
            timer->set(gputime);


 #if 0 //printouts    
    //DEBUGING
    
    std::vector<int4> cli;
    std::vector<float4> clf;

    
    try{
        clf = clf_debug.copyToHost(num);
        cli = cli_debug.copyToHost(num);
    }
    catch (cl::Error er) {
        printf("ERROR(triangle): %s(%s)\n", er.what(), oclErrorString(er.err()));
    }


    int tricount = 0;//count how many particles are colliding with particles
	//for (int i=0; i < num; i++) {  
	for (int i=0; i < num; i++) 
    {  
        if(clf[i].x > 0. or clf[i].y > 0. or clf[i].z > 0.)
        {
		    printf("-----\n");
		    printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
            tricount++;
        }
        /*
        printf("%d particle z %f\n", i, clf[i].w);
        printf("%d triangles tested %d\n", i, cli[i].x);
        printf("%d num particles %d\n", i, cli[i].y);
        printf("%d local size %d\n", i, cli[i].z);
		//printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
        */
    }
    if(tricount >0)
    {
        printf("%d particles collided with a triangle this frame\n", tricount);
        printf("============================================\n");
        printf("***** PRINT triangle collision diagnostics ******\n");
    }

#endif


    }
}

}
