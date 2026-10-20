CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Isrc -fopenmp
LDFLAGS ?= -lm

# ---- backend selection: `make` (CPU) or `make cuda` (NVIDIA GPU) ----
BUILD   ?= cpu
BIN     ?= cncm

SRC := src/main.c src/tensor.c src/kernels.c src/vq.c src/vqvae.c \
       src/ncp.c src/tok.c src/tok_tables.c src/metrics.c src/gpt2.c src/qwen.c src/train.c src/train_tape.c src/train_vq.c

ifeq ($(BUILD),cuda)
ODIR   := build/cuda
CC     := $(NVCC)
CFLAGS := -O2 -std=c++17 -Isrc -DNCM_CUDA \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_86,code=compute_86 \
          -Xcompiler "-Wall -Wextra -fopenmp"
LDFLAGS := -lm -lcudart -lgomp
XFLAG  := -x cu
else
ODIR   := build/cpu
XFLAG  :=
endif

OBJ := $(patsubst src/%.c,$(ODIR)/%.o,$(SRC))

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

$(ODIR)/%.o: src/%.c
	@mkdir -p $(ODIR)
	$(CC) $(XFLAG) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(OBJ:.o=.d)

cuda:
	$(MAKE) BUILD=cuda BIN=cncm-cuda

test: $(BIN)
	python3 tests/test_parity.py --bin ./$(BIN)

tools/gpt2import: tools/gpt2import.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

tools/qwenimport: tools/qwenimport.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

clean:
	rm -rf build $(BIN) cncm-cuda tools/gpt2import tools/qwenimport

.PHONY: all cuda test clean tools/gpt2import tools/qwenimport
