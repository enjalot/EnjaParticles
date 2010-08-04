#include "../enja.h"

void EnjaParticles::loadTriangles(std::vector<Triangle> triangles)
{
    n_triangles = triangles.size();
    //load triangles into cl buffer
    //Triangle is a struct that ends up being 4 float4s
    size_t tri_size = sizeof(Triangle) * n_triangles;
    cl_triangles = cl::Buffer(context, CL_MEM_WRITE_ONLY, tri_size, NULL, &err);
    err = queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, tri_size, &triangles[0], NULL, &event);
    queue.finish();
   
    err = collision_kernel.setArg(2, cl_triangles);   //triangles
    err = collision_kernel.setArg(3, n_triangles);   //number of triangles

	
	size_t sz = n_triangles*sizeof(Triangle);
	printf("sz= %d bytes\n", sz);
    err = collision_kernel.setArg(5, sz, 0);   //number of triangles

    //need to deal with transforms
}

