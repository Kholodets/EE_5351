
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_86,code=\"sm_86\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = spmv
OBJ	        = main.o kernel.o support.o

default: $(EXE)

main.o: main.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

kernel.o: kernel.cu support.h
	$(NVCC) -c -o $@ kernel.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
