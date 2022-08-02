FLAG = -O2 -g -std=c++17
GPUCOMP = /home/jrouze/llvm/build/bin/clang++
GPUFLAG = -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda
GPURUN = LD_LIBRARY_PATH=/home/jrouze/llvm/build/lib


main: SyclGPUs.cpp
	$(GPUCOMP) $(FLAG) $(GPUFLAG) -o $@ $<

clean:
	rm main *.txt

txtclean:
	rm *.txt

gpurun: main
	$(GPURUN) ./main

.PHONY: bench

bench:
	for i in 14 15 16 17; do for j in 0 1 2; do ./main $$i 7 128 1 $$j; done; done
