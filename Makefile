CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Isrc
LDFLAGS ?= -lm

# ---- backend selection: `make` (CPU) or `make cuda` (NVIDIA GPU) ----
BUILD   ?= cpu
BIN     ?= cncm

SRC := src/main.c src/tensor.c src/kernels.c src/vq.c src/vqvae.c \
       src/ncp.c src/tok.c src/metrics.c src/gpt2.c src/train.c

ifeq ($(BUILD),cuda)
ODIR   := build/cuda
CC     := $(NVCC)
CFLAGS := -O2 -std=c++17 -Isrc -DNCM_CUDA \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_86,code=compute_86 \
          -Xcompiler "-Wall -Wextra"
LDFLAGS := -lm -lcudart
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
	$(CC) $(XFLAG) $(CFLAGS) -c $< -o $@

cuda:
	$(MAKE) BUILD=cuda BIN=cncm-cuda

test: $(BIN)
	python3 tests/test_parity.py --bin ./$(BIN)

clean:
	rm -rf build $(BIN) cncm-cuda

.PHONY: all cuda test clean
