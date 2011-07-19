CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(matrix_quat)
set(SD ${CMAKE_SOURCE_DIR})

set(SRC 
	Matrix.cpp
	Point.cpp
	Point1.cpp
	Point2.cpp
	Point3.cpp
	Point4.cpp
	Quat.cpp
)

INCLUDE_DIRECTORIES(.)
ADD_LIBRARY(quatmatpt ${SRC})

