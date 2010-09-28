
# precompile opencl code to .cl file to resolve all includes and preprocessor macros
g++ -E $1.cpp > $1.cl

