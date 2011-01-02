#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <android/log.h>

#include "importgl.h"

#include "util.h"
int deleteVBO(GLuint id)
{
    glBindBuffer(1, id);
    glDeleteBuffers(1, (GLuint*)&id);
    return 1; //success
}

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage)
{
    GLuint id = 0;  // 0 is reserved, glGenBuffersARB() will return non-zero id if success

    glGenBuffers(1, &id);                        // create a vbo
    glBindBuffer(target, id);                    // activate vbo id to use
    glBufferData(target, dataSize, data, usage); // upload data to video card

    // check data size in VBO is same as input array, if not return 0 and delete VBO
    int bufferSize = 0;
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
    if(dataSize != bufferSize)
    {

        __android_log_print(ANDROID_LOG_INFO, "RTPS", "vbo prob: %d", id);
        glDeleteBuffers(1, &id);
        id = 0;
        //cout << "[createVBO()] Data size is mismatch with input array\n";
        //printf("[createVB()] Data size is mismatch with input array\n");
    }
    //this was important for working inside blender!
    glBindBuffer(target, 0);

    return id;      // return VBO id
}

