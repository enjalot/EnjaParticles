LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_CPP_EXTENSION := cpp
LOCAL_MODULE := rtps

LOCAL_CFLAGS := -DANDROID_NDK \
                -DDISABLE_IMPORTGL

LOCAL_SRC_FILES := \
    importgl.c \
	glue.c \
    rtps.cpp \
	rtpslib/RTPS.cpp \
	rtpslib/RTPSettings.cpp \
	rtpslib/structs.cpp \
	rtpslib/util.cpp \
	rtpslib/render/Render.cpp \
	rtpslib/domain/Domain.cpp \
	rtpslib/domain/IV.cpp \
	rtpslib/system/Simple.cpp \
	rtpslib/system/simple/Euler.cpp \
	rtpslib/system/SPH.cpp \
	rtpslib/system/sph/Euler.cpp \
	rtpslib/system/sph/LeapFrog.cpp \
	rtpslib/system/sph/Density.cpp \
	rtpslib/system/sph/Pressure.cpp \
	rtpslib/system/sph/Viscosity.cpp \
	rtpslib/system/sph/XSPH.cpp \
	rtpslib/system/sph/Collision_wall.cpp \
    app-android.c \

LOCAL_LDLIBS := -lGLESv1_CM -ldl -llog

include $(BUILD_SHARED_LIBRARY)
