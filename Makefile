CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Isrc -fopenmp
LDFLAGS ?= -lm

# ---- backend selection: `make` (CPU) or `make cuda` (NVIDIA GPU) ----
BUILD   ?= cpu
BIN     ?= cncm

SRC := src/main.c src/tensor.c src/kernels.c src/vq.c src/vqvae.c \
       src/ncp.c src/tok.c src/tok_tables.c src/metrics.c src/gpt2.c src/qwen.c src/train.c src/train_tape.c src/train_vq.c src/ckpt.c

ifeq ($(BUILD),cuda)
ODIR   := build/cuda
CC     := $(NVCC)
CFLAGS := -O2 -std=c++17 -Isrc -DNCM_CUDA \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_86,code=compute_86 \
          -Xcompiler "-Wall -Wextra -fopenmp"
LDFLAGS := -lm -lcudart -lcublas -lgomp
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

# Test binaries link the full CPU object set (minus main); gcheck and
# ckpt_unit are synthetic and need no weights.
CPU_OBJ := $(patsubst src/%.c,build/cpu/%.o,$(SRC))
CPU_TESTOBJ := $(filter-out build/cpu/main.o,$(CPU_OBJ))
tests/gcheck: tests/gcheck.c $(CPU_TESTOBJ)
	$(CC) $(CFLAGS) -o $@ tests/gcheck.c $(CPU_TESTOBJ) $(LDFLAGS)
tests/ckpt_unit: tests/ckpt_unit.c $(CPU_TESTOBJ)
	$(CC) $(CFLAGS) -o $@ tests/ckpt_unit.c $(CPU_TESTOBJ) $(LDFLAGS)

test: cncm tests/gcheck tests/ckpt_unit
	./tests/test_units.sh
	./tests/test_ckpt.sh --bin ./cncm --quick
test-resume: cncm
	./tests/test_ckpt.sh --bin ./cncm
test-resume-cuda: cncm-cuda
	./tests/test_ckpt.sh --bin ./cncm-cuda --cuda-tol 1e-3

tools/gpt2import: tools/gpt2import.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

tools/qwenimport: tools/qwenimport.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

tools/export_text: tools/export_text.c src/tok.c src/tok_tables.c src/tensor.c
	$(CC) -O2 -std=c11 -Wall -Wextra -Isrc -o $@ tools/export_text.c src/tok.c src/tok_tables.c src/tensor.c -lm

clean:
	rm -rf build $(BIN) cncm-cuda tools/gpt2import tools/qwenimport tools/export_text tests/gcheck tests/ckpt_unit

.PHONY: all cuda test test-resume test-resume-cuda clean tools/gpt2import tools/qwenimport tools/export_text
