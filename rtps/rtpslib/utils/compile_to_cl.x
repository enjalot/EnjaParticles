
# precompile opencl code to .cl file to resolve all includes and preprocessor macros
g++ -E -x c++ $1 > $2/$1

