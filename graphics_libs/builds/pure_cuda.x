rm pure_cuda/C*C*txt
rm ../C*C*txt
#(cd pure_cuda; cmake --trace -DPURE_CUDA=ON ../..)
(cd pure_cuda; cmake -DPURE_CUDA=ON ../..)
