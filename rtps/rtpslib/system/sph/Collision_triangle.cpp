#include "../SPH.h"

namespace rtps {

void SPH::loadCollision_tri()
{
    triangles_loaded = false;
    printf("create collision wall kernel\n");

    std::string path(SPH_CL_SOURCE_DIR);
    path += "/collision_tri_cl.cl";
    k_collision_tri = Kernel(ps->cli, path, "collision_triangle");
  
    k_collision_tri.setArg(0, cl_vars_sorted.getDevicePtr());
    // 1 = triangles
    // 2 = n_triangles
    k_collision_tri.setArg(3, ps->settings.dt);
    k_collision_tri.setArg(4, cl_SPHParams.getDevicePtr());
    // 5 = local triangles

} 


void SPH::loadTriangles(std::vector<Triangle> triangles)
{
    int n_triangles = triangles.size();
    printf("n triangles: %d\n", n_triangles);
    //load triangles into cl buffer
    //Triangle is a struct that ends up being 4 float4s
    cl_triangles = Buffer<Triangle>(ps->cli, triangles);
   
    k_collision_tri.setArg(1, cl_triangles.getDevicePtr());     //triangles
    k_collision_tri.setArg(2, n_triangles);                     //number of triangles

    //printf("sizeof(Triangle) = %d\n", (int) sizeof(Triangle));


    //TODO: get local mem size from opencl
    size_t max_loc_memory = 1024 << 4;  // 16k bytes local memory on mac
    int max_tri = max_loc_memory / sizeof(Triangle);
    //max_tri = n_triangles;
    max_tri = 220; // fits in cache
    printf("max_tri= %d\n", max_tri);

    size_t sz = max_tri*sizeof(Triangle);
    printf("sz= %zd bytes\n", sz);

    k_collision_tri.setArgShared(5, sz);


    triangles_loaded = true;
    //exit(0);

}

void SPH::collide_triangles()
{
    
    if(triangles_loaded)
    {
        //printf("execute!\n");
        k_collision_tri.execute(num, 128);
    }
}

}
