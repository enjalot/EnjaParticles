#include <stdlib.h>
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
    err = collision_kernel.setArg(5, sz, 0);   //number of triangles
	//exit(0);
#endif

    //need to deal with transforms
}
//----------------------------------------------------------------------
void EnjaParticles::loadBoxes(std::vector<Box> boxes, std::vector<int> tri_offsets)
{
    n_boxes = boxes.size();
    //load boxes into cl buffer
    //Box is a struct that ends up being 6 floats
    size_t box_size = sizeof(Box) * n_boxes;
    cl_boxes = cl::Buffer(context, CL_MEM_WRITE_ONLY, box_size, NULL, &err);
    err = queue.enqueueWriteBuffer(cl_boxes, CL_TRUE, 0, box_size, &boxes[0], NULL, &event);
    queue.finish();
   
    err = collision_kernel.setArg(2, cl_boxes);   //boxes
    err = collision_kernel.setArg(3, n_boxes);   //number of boxes

	printf("sizeof(Box) = %d\n", (int) sizeof(Box));

	size_t offset_size = sizeof(int)*tri_offsets.size();
	cl_tri_offsets = cl::Buffer(context, CL_MEM_WRITE_ONLY, offset_size, NULL, &err);
    err = queue.enqueueWriteBuffer(cl_tri_offsets, CL_TRUE, 0, offset_size, &tri_offsets[0], NULL, &event);
    err = collision_kernel.setArg(5, cl_tri_offsets);   //number of boxes
	queue.finish();

#ifdef OPENCL_SHARED

	size_t max_loc_memory = 1024 << 4;  // 16k bytes local memory on mac
	int max_box = max_loc_memory / sizeof(Box);
	//max_tri = n_triangles;
	max_box = 600; // fits in cache
	printf("max_box= %d\n", max_box);
	
	size_t sz = max_box*sizeof(Box);
	printf("sz= %d bytes\n", sz);

   // experimenting with hardcoded local memory in collision_ge.cl
    err = collision_kernel.setArg(6, sz, 0);   //number of boxes
	//exit(0);
#endif

    //need to deal with transforms
}

