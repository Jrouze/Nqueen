FLAG = -O2 -g -std=c++17
GPUCOMP = /home/jrouze/llvm/build/bin/clang++
GPUFLAG = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
GPURUN = LD_LIBRARY_PATH=/home/jrouze/llvm/build/lib
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jrouze/llvm/build/lib
# LD_LIBRARY_PATH=/home/jrouze/llvm/build/lib/:/home/jrouze/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin/

main: multi_gpu_queens.dp.cpp
	$(GPUCOMP) $(FLAG) $(GPUFLAG) -fopenmp=libiomp5 -o $@ $<

# sycl_multi: SyclGPUs.cpp
# 	$(GPUCOMP) $(FLAG) $(GPUFLAG) -o $@ $<

cuda_multi: multi_gpu_queens.cu
	nvcc -O2 multi_gpu_queens.cu -Xcompiler -fopenmp -o $@

cuda_solo: queens.cu
	nvcc -O2 -o $@ $<

clean:
	rm sycl_multi cuda_multi cuda_solo main *.txt

txtclean:
	rm *.txt

gpurun: sycl_multi
	$(GPURUN) ./sycl_multi


.PHONY: bench

bench:
	for i in 15 16 17 18; do ./main $$i 7 128; done
