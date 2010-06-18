/* San Angeles Observation OpenGL ES version example
 * Copyright 2009 The Android Open Source Project
 * All rights reserved.
 *
 * This source is free software; you can redistribute it and/or
 * modify it under the terms of EITHER:
 *   (1) The GNU Lesser General Public License as published by the Free
 *       Software Foundation; either version 2.1 of the License, or (at
 *       your option) any later version. The text of the GNU Lesser
 *       General Public License is included with this source in the
 *       file LICENSE-LGPL.txt.
 *   (2) The BSD-style license that is included with this source in
 *       the file LICENSE-BSD.txt.
 *
 * This source is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 * LICENSE-LGPL.txt and LICENSE-BSD.txt for more details.
 */
#include <jni.h>
#include <sys/time.h>
#include <time.h>
#include <android/log.h>
#include <stdint.h>

int   gAppAlive   = 1;

static int  sWindowWidth  = 320;
static int  sWindowHeight = 480;
static int  sDemoStopped  = 0;
static long sTimeOffset   = 0;
static int  sTimeOffsetInit = 0;
static long sTimeStopped  = 0;

//hack so we can pass floats...
static float tx = 0;
static float ty = 0;
static float mx = 0;
static float my = 0;
static float dx = 0;
static float dy = 0;

static long
_getTime(void)
{
    struct timeval  now;

    gettimeofday(&now, NULL);
    return (long)(now.tv_sec*1000 + now.tv_usec/1000);
}

/* Call to initialize the graphics state */
void
Java_com_enja_particles_EnjRenderer_nativeInit( JNIEnv*  env )
{
    importGLInit();
    appInit();
    gAppAlive    = 1;
    sDemoStopped = 0;
    sTimeOffsetInit = 0;
    float b = 1.f;
    appTest(1, &b);
}

void
Java_com_enja_particles_EnjRenderer_nativeResize( JNIEnv*  env, jobject  thiz, jint w, jint h )
{
    sWindowWidth  = w;
    sWindowHeight = h;
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "resize w=%d h=%d", w, h);
}

/* Call to finalize the graphics state */
void
Java_com_enja_particles_EnjRenderer_nativeDone( JNIEnv*  env )
{
    appDeinit();
    importGLDeinit();
}

/* This is called to indicate to the render loop that it should
 * stop as soon as possible.
 */
void
Java_com_enja_particles_EnjGLSurfaceView_nativePause( JNIEnv*  env )
{
    sDemoStopped = !sDemoStopped;
    if (sDemoStopped) {
        /* we paused the animation, so store the current
         * time in sTimeStopped for future nativeRender calls */
        sTimeStopped = _getTime();
    } else {
        /* we resumed the animation, so adjust the time offset
         * to take care of the pause interval. */
        sTimeOffset -= _getTime() - sTimeStopped;
    }
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "native pause called sDemoStoppe =%d", sDemoStopped);
}

/* Call to render the next GL frame */
void
Java_com_enja_particles_EnjRenderer_nativeRender( JNIEnv*  env )
{
    long   curTime;

    /* NOTE: if sDemoStopped is TRUE, then we re-render the same frame
     *       on each iteration.
     */
    if (sDemoStopped) {
        curTime = sTimeStopped + sTimeOffset;
    } else {
        curTime = _getTime() + sTimeOffset;
        if (sTimeOffsetInit == 0) {
            sTimeOffsetInit = 1;
            sTimeOffset     = -curTime;
            curTime         = 0;
        }
    }

    //__android_log_print(ANDROID_LOG_INFO, "SanAngeles", "curTime=%ld", curTime);

    appRender(curTime, sWindowWidth, sWindowHeight);
}

/* call to set new spawn point from a touch */
void
Java_com_enja_particles_EnjGLSurfaceView_nativeTouch( JNIEnv*  env, jobject thiz, jfloat x, jfloat y )
{
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "native touch x=%f y=%f", x, y);
    tx = (float)x;
    ty = (float)y;
    appTouch(&tx,&ty);
}

/* call to set finger position from touch */
void
Java_com_enja_particles_EnjGLSurfaceView_nativeDown( JNIEnv*  env, jobject thiz, jfloat x, jfloat y )
{
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "touch x=%f y=%f", x, y);
    //appDown(x, y);
    dx = (float)x;
    dy = (float)y;
    appDown(&dx,&dy);
}


/* call to rotate from a touch move */
void
Java_com_enja_particles_EnjGLSurfaceView_nativeMove( JNIEnv*  env, jobject thiz, jfloat x, jfloat y )
{
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "move x=%f y=%f", x, y);
    //appMove(x, y);
    mx = (float)x;
    my = (float)y;
    appMove(&mx,&my);
}


