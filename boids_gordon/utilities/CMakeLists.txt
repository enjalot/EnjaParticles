cmake_minimum_required(VERSION 2.6)

project(utilities)
set(utils Array3D.cpp Vec3i.cpp Vec3.cpp CVec3.cpp matrix3x3f.cpp )

include_directories( . )
add_library(utilities STATIC ${utils})

#ADD_DEFINITIONS(-arch x86_64)

# for mac
#ADD_DEFINITIONS(-arch i386)

#add_executable(a.out ${utils})
#target_link_libraries(cpu ${libraries})
