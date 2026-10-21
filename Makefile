CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Isrc -fopenmp
LDFLAGS ?= -lm

# ---- backend selection: `make` (CPU) or `make cuda` (NVIDIA GPU) ----
BUILD   ?= cpu
BIN     ?= cncm

SRC := src/main.c src/tensor.c src/kernels.c src/vq.c src/vqvae.c \
       src/ncp.c src/tok.c src/tok_tables.c src/metrics.c src/gpt2.c src/qwen.c src/train.c src/train_tape.c src/train_vq.c src/train_qwen.c src/viewer.c src/ckpt.c

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

# SDL2 is optional: viewer.c compiles to no-ops without it. This block
# must stay after the BUILD selection above (the cuda leg reassigns flags).
SDL_CFLAGS := $(shell sdl2-config --cflags 2>/dev/null)
SDL_LIBS   := $(shell sdl2-config --libs 2>/dev/null)
ifneq ($(SDL_CFLAGS),)
CFLAGS += $(SDL_CFLAGS) -DCNCM_HAVE_SDL2
LDFLAGS += $(SDL_LIBS)
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
tests/fontcheck: tests/fontcheck.c build/cpu/viewer.o
	$(CC) $(CFLAGS) -o $@ tests/fontcheck.c build/cpu/viewer.o $(SDL_LIBS) -lm

test: cncm tests/gcheck tests/ckpt_unit
	./tests/test_units.sh
	./tests/test_ckpt.sh --bin ./cncm --quick
test-resume: cncm
	./tests/test_ckpt.sh --bin ./cncm
test-resume-cuda: cncm-cuda
	./tests/test_ckpt.sh --bin ./cncm-cuda --cuda-tol 1e-3
test-viewer: cncm tests/fontcheck
	./tests/test_viewer.sh --bin ./cncm
test-viewer-cuda: cncm-cuda tests/fontcheck
	./tests/test_viewer.sh --bin ./cncm-cuda

tools/gpt2import: tools/gpt2import.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

tools/qwenimport: tools/qwenimport.c
	$(CC) -O2 -std=c11 -Wall -Wextra -o $@ $<

tools/export_text: tools/export_text.c src/tok.c src/tok_tables.c src/tensor.c
	$(CC) -O2 -std=c11 -Wall -Wextra -Isrc -o $@ tools/export_text.c src/tok.c src/tok_tables.c src/tensor.c -lm

clean:
	rm -rf build $(BIN) cncm-cuda tools/gpt2import tools/qwenimport tools/export_text tests/gcheck tests/ckpt_unit tests/fontcheck

.PHONY: all cuda test test-resume test-resume-cuda clean tools/gpt2import tools/qwenimport tools/export_text
