
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_86,code=\"sm_86\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = vector_reduction
OBJ	        = vector_reduction_cu.o vector_reduction_cpp.o vector_reduction_kernel_cu.o

default: $(EXE)

vector_reduction_cu.o: vector_reduction.cu
	$(NVCC) -c -o $@ vector_reduction.cu $(NVCC_FLAGS)

vector_reduction_kernel_cu.o: vector_reduction_kernel.cu
	$(NVCC) -c -o $@ vector_reduction_kernel.cu $(NVCC_FLAGS)

vector_reduction_cpp.o: vector_reduction_gold.cpp
	$(NVCC) -c -o $@ vector_reduction_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
