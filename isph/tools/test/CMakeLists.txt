SET (CPPFILES 
	main.cpp
)
#testsort.cpp


message(" **** ${isph_SOURCE_DIR}/isphlib")
message(" *** ${OPENCL_LIBRARIES}")

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${isph_SOURCE_DIR}/isphlib
    ${OPENGL_INCLUDE_DIR}
)


LINK_DIRECTORIES(
	${isph_BINARY_DIR}/isphlib
)

ADD_EXECUTABLE(main.x ${CPPFILES})

#ADD_DEFINITIONS(-m32)
message("++++ sph: ${SPH}")


TARGET_LINK_LIBRARIES(main.x
   ${SPH}
   ${OPENCL_LIBRARIES}
)


