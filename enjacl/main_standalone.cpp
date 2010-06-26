#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <string.h>
#include <string>

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #include <GLUT/glut.h>
    #include <OpenGL/CGLCurrent.h> //is this really necessary?
    //OpenCL stuff
    #include <OpenCL/opencl.h>
    #include <OpenCL/cl_gl.h>
    #include <OpenCL/cl_gl_ext.h>
    #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
    //OpenGL stuff
    #include <GL/glx.h>
    #include <GL/glut.h>
    //OpenCL stuff
    #include <CL/opencl.h>
    #include <CL/cl_gl.h>
    #include <CL/cl_gl_ext.h>
    #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#include "util.h"

//particle stuff
#define NUM_PARTICLES 1
Vec4 vertices[NUM_PARTICLES];

GLuint vbo = 0;
unsigned int vbo_size = sizeof(Vec4) * NUM_PARTICLES;

int window_width = 400;
int window_height = 300;
int glutWindowHandle = 0;
float translate_z = -3.f;

void init_gl();

void appKeyboard(unsigned char key, int x, int y);
void appRender();
void appDestroy();

//opencl stuff
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_mem vbo_cl;
cl_program cpProgram;
cl_int ciErrNum;
size_t szGlobalWorkSize[] = {NUM_PARTICLES};


void init_cl();
void runKernel();

void init_cl()
{
    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get device requested on command line, if any
    unsigned int uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;

    bool bSharingSupported = false;
    for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
    {
        size_t extensionSize;
        ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        if(extensionSize > 0) 
        {
            char* extensions = (char*)malloc(extensionSize);
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
            //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            std::string stdDevString(extensions);
            free(extensions);

            size_t szOldPos = 0;
            size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
            while (szSpacePos != stdDevString.npos)
            {
                if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                {
                    // Device supports context sharing with OpenGL
                    uiDeviceUsed = i;
                    bSharingSupported = true;
                    break;
                }
                do 
                {
                    szOldPos = szSpacePos + 1;
                    szSpacePos = stdDevString.find(' ', szOldPos);
                } 
                while (szSpacePos == szOldPos);
            }
        }
    }

    // Define OS-specific context properties and create the OpenCL context
    //#if defined (__APPLE_CC__)
    #if defined (__APPLE__) || defined(MACOSX)
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
            0
        };
        cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
    #else
        #if defined WIN32 // Win32
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                0
            };
            cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
        #else
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                0
            };
            cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
        #endif
    #endif
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log device used (reconciled for requested requested and/or CL-GL interop capable devices, as applies)
    //shrLog("Device # %u, ", uiDeviceUsed);
    //oclPrintDevName(LOGBOTH, cdDevices[uiDeviceUsed]);
    
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    int pl;
    size_t program_length;
    printf("open the program\n");
    //char* cSourceCL = file_contents("enja.cl", &pl);
    
    std::string path(CL_SOURCE_DIR);
    path += "/enja.cl";
    //printf("%s\n", path.c_str());
    char* cSourceCL = file_contents(path.c_str(), &pl);
    //char* cSourceCL = file_contents("/panfs/panasas1/users/idj03/research/iansvn/enjacl/build/enja.cl", &pl);
    //printf("file: %s\n", cSourceCL);
    program_length = (size_t)pl;
    //shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
                      (const char **) &cSourceCL, &program_length, &ciErrNum);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    printf("building the program\n");
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    //ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("houston we have a problem\n%s\n", oclErrorString(ciErrNum));
    }

    printf("program built\n");
    ckKernel = clCreateKernel(cpProgram, "enja", &ciErrNum);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    printf("kernel made: %s\n", oclErrorString(ciErrNum));



}



void init_gl()
{
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
/*
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
*/
    return;

}

void appKeyboard(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            appDestroy();
            break;
    }
}

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage)
{
    GLuint id = 0;  // 0 is reserved, glGenBuffersARB() will return non-zero id if success

    glGenBuffers(1, &id);                        // create a vbo
    glBindBuffer(target, id);                    // activate vbo id to use
    //glBufferData(target, dataSize, data, usage); // upload data to video card
    glBufferData(GL_ARRAY_BUFFER, vbo_size, vertices, GL_DYNAMIC_DRAW); // upload data to video card

    // check data size in VBO is same as input array, if not return 0 and delete VBO
    int bufferSize = 0;
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
    if(dataSize != bufferSize)
    {
        glDeleteBuffers(1, &id);
        id = 0;
        //cout << "[createVBO()] Data size is mismatch with input array\n";
        printf("[createVBO90] Data size is mismatch with input array\n");
    }

    return id;      // return VBO id
}


void appRender()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //printf("render!\n");
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH); 
    
    //update the buffers with new vertices and colors
    
    //glBindBuffer(GL_ARRAY_BUFFER, glres.vertex_buffer);
    //glBindBuffer(GL_ARRAY_BUFFER, glres.color_buffer);
    //glBufferData(GL_ARRAY_BUFFER, glres.vbo_size, colors, GL_DYNAMIC_DRAW);
    //glColorPointer(4, GL_FLOAT, 0, 0);

    runKernel();

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBufferData(GL_ARRAY_BUFFER, vbo_size, vertices, GL_DYNAMIC_DRAW);
    glVertexPointer(4, GL_FLOAT, 0, 0);
/*
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    Vec4 tmp = ((Vec4*)ptr)[0];
    printf("VBO coord[0] %g %g %g\n", tmp.x, tmp.y, tmp.z);
    glUnmapBufferARB(GL_ARRAY_BUFFER);
*/

    glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    glColor3f(0,1,0);
    glPointSize(10.);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
//    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);



    glutSwapBuffers();
    glutPostRedisplay();
}

void appDestroy()
{
    
    if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(vbo)
    {
        glBindBuffer(1, vbo);
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    if(vbo_cl)clReleaseMemObject(vbo_cl);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
    if(cdDevices)delete(cdDevices);

    if(glutWindowHandle)glutDestroyWindow(glutWindowHandle);

    exit(0);
}

void runKernel()
{
    ciErrNum = CL_SUCCESS;
 
#ifdef GL_INTEROP   
    // map OpenGL buffer object for writing from OpenCL
    glFinish();
    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#endif

    float anim = 0.0001;
    ciErrNum = clSetKernelArg(ckKernel, 1, sizeof(float), &anim);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, szGlobalWorkSize, NULL, 0,0,0 );
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);
#else

    // Explicit Copy 
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_ARRAY_BUFFER, vbo);    

    // map the buffer object into client's memory
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);

    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, vbo_size, ptr, 0, NULL, NULL);
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    glUnmapBufferARB(GL_ARRAY_BUFFER); 
#endif
}


int main(int argc, char** argv)
{

    //initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);

    glutWindowHandle = glutCreateWindow("EnjaParticles");

    glutDisplayFunc(appRender); //main rendering function
    glutKeyboardFunc(appKeyboard);
/*
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);
*/

    // initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    printf("GLEW supported?: %d\n", bGLEW);

    //initialize the OpenGL scene for rendering
    init_gl();
    
    vertices[0].x = 0.f;
    vertices[0].y = 0.f;
    vertices[0].z = 0.f;
    vertices[0].w = 1.f;

    vbo = createVBO(vertices, vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    
    init_cl();

     #ifdef GL_INTEROP
        printf("gl interop!\n");
        // create OpenCL buffer from GL VBO
        vbo_cl = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, vbo, &ciErrNum);
        //printf("wwwwttttffff\n");
        //printf("SUCCES?: %s\n", oclErrorString(ciErrNum));
    #else
        printf("no gl interop!\n");
        // create standard OpenCL mem buffer
        vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, vbo_size, NULL, &ciErrNum);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, vbo_size, vertices, 0, NULL, NULL);
        clFinish(cqCommandQueue);
    #endif

    ciErrNum  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &vbo_cl);


    glutMainLoop();
    
    printf("doesn't happen does it\n");
    appDestroy();
    return 0;
}


























