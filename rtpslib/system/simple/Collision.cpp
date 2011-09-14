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


void EnjaParticles::loadTriangles(std::vector<Triangle> triangles)
{
    printf("LOAD TRIANGLES collision: %d\n", collision);
    if (!collision)
        return;
    n_triangles = triangles.size();
    printf("n triangles: %d\n", n_triangles);
    //load triangles into cl buffer
    //Triangle is a struct that ends up being 4 float4s
    size_t tri_size = sizeof(Triangle) * n_triangles;
    cl_triangles = cl::Buffer(context, CL_MEM_WRITE_ONLY, tri_size, NULL, &err);
    err = queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, tri_size, &triangles[0], NULL, &event);
    queue.finish();
   
    err = collision_kernel.setArg(2, cl_triangles);   //triangles
    err = collision_kernel.setArg(3, n_triangles);   //number of triangles

    printf("sizeof(Triangle) = %d\n", (int) sizeof(Triangle));



#ifdef OPENCL_SHARED

    size_t max_loc_memory = 1024 << 4;  // 16k bytes local memory on mac
    int max_tri = max_loc_memory / sizeof(Triangle);
    //max_tri = n_triangles;
    max_tri = 220; // fits in cache
    printf("max_tri= %d\n", max_tri);
        
    size_t sz = max_tri*sizeof(Triangle);
    printf("sz= %d bytes\n", sz);

   // experimenting with hardcoded local memory in collision_ge.cl
    err = collision_kernel.setArg(6, sz, 0);   //number of triangles
    //exit(0);
#endif  

    //need to deal with transforms
}

