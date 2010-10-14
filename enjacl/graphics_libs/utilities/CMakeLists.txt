cmake_minimum_required(VERSION 2.6)

project(utilities)
set(utils Array3D.cpp Vec3i.cpp Vec3.cpp CVec3.cpp matrix3x3f.cpp )

ADD_LIBRARY(utilities STATIC ${utils})


