#include "../OUTER.h"

namespace rtps {
namespace outer {

    CollisionTriangle::CollisionTriangle(std::string path, CL* cli_, EB::Timer* timer_, int max_triangles)
    {
        cli = cli_;
        timer = timer_;
        std::vector<Triangle> maxtri(max_triangles);
        cl_triangles = Buffer<Triangle>(cli, maxtri);


        triangles_loaded = false;
        printf("create collision wall kernel\n");
        path += "/collision_tri.cl";
        k_collision_tri = Kernel(cli, path, "collision_triangle");
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
                                    Buffer<OUTERParams>& sphp,
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

    //TODO: avoid need for this function?
    void OUTER::loadTriangles(std::vector<Triangle> &triangles)
    {
        collision_tri.loadTriangles(triangles);
    }

}
