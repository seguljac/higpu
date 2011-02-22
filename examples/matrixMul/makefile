
GCC=g++
CFLAGS=-Wall -O3 -m32
LDFLAGS=-m32 -L/usr/local/cuda/lib -lcudart

NVCC=nvcc
NVCC_FLAGS+=-O3 -DUNIX -m32
NVCC_FLAGS+= -I/usr/local/cuda/include

HICUDA=hicuda
HICUDA_OPT=-m32

######################################################################

PROG_BASE=mm
PROG_SEQ=$(PROG_BASE)_seq
PROG_CUDA=$(PROG_BASE)_cuda
PROG_HICUDA=$(PROG_BASE)_hicuda
# compiling the hicuda program, ignoring the hicuda directives
PROG_HICUDA_SEQ=$(PROG_BASE)_hicuda_seq

PROGS=$(PROG_HICUDA_SEQ) $(PROG_CUDA) $(PROG_HICUDA)

default: $(PROG_HICUDA).cuda/$(PROG_HICUDA).cu

all: $(PROGS)

######################################################################

$(PROG_CUDA): $(PROG_CUDA).cu_o common.o
	$(GCC) $(LDFLAGS) -o $@ $^

$(PROG_CUDA).cu_o: $(PROG_CUDA).cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

######################################################################

$(PROG_HICUDA): $(PROG_HICUDA).cuda/$(PROG_HICUDA).cu
	cd $(PROG_HICUDA).cuda; $(NVCC) $(NVCC_FLAGS) -o $@ $(PROG_HICUDA).cu
	cp $(PROG_HICUDA).cuda/$(PROG_HICUDA) .

$(PROG_HICUDA).cuda/$(PROG_HICUDA).cu: $(PROG_HICUDA).c common.c
	hicuda -o $(PROG_HICUDA) $(HICUDA_OPT) $^

######################################################################

$(PROG_HICUDA_SEQ): $(PROG_HICUDA).o common.o
	$(GCC) $(LDFLAGS) -o $@ $^

######################################################################

$(PROG_SEQ): $(PROG_SEQ).o common.o
	$(GCC) -o $@ $^

######################################################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $<

%.o: %.cpp
	$(GCC) $(CFLAGS) -c $<

######################################################################

clean:
	rm -f $(PROGS) *.o *.cu_o *.linkinfo
	rm -rf $(PROG_HICUDA).cuda*

clean-all: clean
	rm -rf *.t *.i *.B *.ipakeep

######################################################################

%.cu: ;

.PHONY: default all clean

