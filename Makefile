CC = gcc
NVCC = nvcc
CFLAGS = -lm

all: boxfilter_cpu boxfilter_gpu laplacian_cpu laplacian_gpu

boxfilter_cpu: boxfilter/box_filter.c
	$(CC) $^ -o boxfilter/boxfilterCpu $(CFLAGS)
	./boxfilter/boxfilterCpu > output

boxfilter_gpu: boxfilter/boxfilter.cu
	$(NVCC) $^ -o boxfilter/boxfilterGPU
	./boxfilter/boxfilterGPU >> output

laplacian_cpu: laplacian/laplacian.c
	$(CC) $^ -o laplacian/laplacianCpu $(CFLAGS)
	./laplacian/laplacianCpu >> output

laplacian_gpu: laplacian/laplacian.cu
	$(NVCC) $^ -o laplacian/laplacianGPU
	./laplacian/laplacianGPU >> output

clean:
	rm -f output boxfilter/boxfilterCpu boxfilter/boxfilterGPU laplacian/laplacianCpu laplacian/laplacianGPU
