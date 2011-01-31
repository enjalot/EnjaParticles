#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <GL/glew.h>

#include "Swarm.h"

namespace rtps{

Swarm::Swarm(RTPS *rtps, int n){
	// number of boids
	num = n;

	// system
	ps = rtps;

	// resize vectos
	positions.resize(num);
	velocities.resize(num);
	colors.resize(num);

 	// boids settings
        this->maxspeed = ps->getRTPSettings().maxspeed;
        this->separationdist = ps->getRTPSettings().separationdist;
        this->searchradius = ps->getRTPSettings().searchradius;
        this->color = ps->getRTPSettings().color;

	
	srand(time(0));

	// filling the data with initial values
	for(int i=0; i < num; i++){
		positions[i] = float4((float)rand()/RAND_MAX, (float)rand()/RAND_MAX, (float)rand()/RAND_MAX, 1.f);
	    	velocities[i] = float4((float)rand()/RAND_MAX, (float)rand()/RAND_MAX, (float)rand()/RAND_MAX, 1.f);
		//printf("%d=> %f, %f, %f\n", i, positions[i].x, positions[i].y, positions[i].z);
		colors[i] = color;
	}
	//std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));	
	//std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 1.0f));	

	// setting the VBOs
	managed = true;
    	pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    	col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	// bind the buffer with the VBO
	cl_position = rtps::Buffer<float4>(ps->cli, pos_vbo);
    	cl_color = rtps::Buffer<float4>(ps->cli, col_vbo);
}
        
Swarm::~Swarm(){
  	if(pos_vbo && managed){
        	glBindBuffer(1, pos_vbo);
        	glDeleteBuffers(1, (GLuint*)&pos_vbo);
        	pos_vbo = 0;
    	}
    	if(col_vbo && managed){
        	glBindBuffer(1, col_vbo);
        	glDeleteBuffers(1, (GLuint*)&col_vbo);
        	col_vbo = 0;
    	}
}

void Swarm::update(){
#ifdef CPU
    FlockIt_CPU();
    	
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
		
    //printf("%d=> %f, %f, %f\n", 0, positions[0].x, positions[0].y, positions[0].z);
#endif
#ifdef GPU
	printf("GPU implementation COMING SOON!!!!\n");
#endif
}

}


