
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_86,code=\"sm_86,compute_86\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = vectoradd
OBJ	        = vectoradd_cu.o vectoradd_kernel_cu.o vectoradd_gold.o

default: $(EXE)

vectoradd_cu.o: vectoradd.cu vectoradd_kernel.cu vectoradd.h
	$(NVCC) -c -o $@ vectoradd.cu $(NVCC_FLAGS)

vectoradd_kernel_cu.o: vectoradd.cu vectoradd_kernel.cu vectoradd.h
	$(NVCC) -c -o $@ vectoradd_kernel.cu $(NVCC_FLAGS)

vectoradd_gold.o: vectoradd_gold.cpp
	$(NVCC) -c -o $@ vectoradd_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
