# Pure-C Wan 2.1 T2V client (1.3B scaffold)
#
#   make -C x/wan-c
#   UMA_WAN_LOCAL=1 ./wan-cli --ckpt-dir ... --prompt "..." --out out.mp4
#
# Links libuma_client from bmtl uma_toolkit (broker protocol only).
# Host kernels: compiles uma_wan_ops.c from the toolkit tree.

BMTL_ROOT ?= $(abspath $(CURDIR)/../../../bmtl)
BMTL_UMA ?= $(BMTL_ROOT)/targets/m4-max/hardware/uma_toolkit
BMTL_UMA_ALT ?= $(BMTL_ROOT)/hardware_lab/lanes/m4/uma_toolkit

SDK   = $(shell xcrun --sdk macosx --show-sdk-path)
CLANG = $(shell xcrun -f clang)

WAN_SRCS = \
	main.c \
	wan_ctx.c \
	wan_config.c \
	gguf_min.c \
	safetensors_min.c \
	zip_weight.c \
	uma_buf_load.c \
	sched_unipc.c \
	tokenizer_spm.c \
	encode_mp4.c \
	wan_graph.c \
	wan_weights.c \
	wan_profile.c \
	pipeline_t2v.c \
	t5_umt5.c \
	dit_wan.c \
	vae_wan.c

WAN_OBJS = $(WAN_SRCS:.c=.o)

TEST_SRCS = tests/test_sched_unipc.c tests/test_validate.c tests/test_gguf_roundtrip.c tests/test_ops_wire.c tests/test_tokenizer_spm.c
TEST_BINS = tests/test_sched_unipc tests/test_validate tests/test_gguf_roundtrip tests/test_ops_wire tests/test_tokenizer_spm

# Resolve toolkit path (prefer primary).
BMTL_UMA_RESOLVED := $(if $(wildcard $(BMTL_UMA)/src/uma_client.c),$(BMTL_UMA),$(BMTL_UMA_ALT))
UMA_WAN_OPS_C := $(BMTL_UMA_RESOLVED)/src/uma_wan_ops.c
UMA_WAN_OPS_O := uma_wan_ops.o

CFLAGS = -O2 -std=c11 -Wall -Wextra -isysroot $(SDK) \
	-I$(CURDIR)/include -I$(BMTL_UMA_RESOLVED)/include -I$(BMTL_UMA_RESOLVED)/src -fPIC

LDFLAGS = -L$(BMTL_UMA_RESOLVED)/src -luma_client -lm -framework Accelerate

.PHONY: all clean test check-paths

all: check-paths wan-cli

check-paths:
	@if [ -f "$(BMTL_UMA)/src/uma_client.c" ]; then \
	  echo "BMTL_UMA=$(BMTL_UMA)"; \
	elif [ -f "$(BMTL_UMA_ALT)/src/uma_client.c" ]; then \
	  echo "BMTL_UMA=$(BMTL_UMA_ALT)"; \
	else \
	  echo "missing uma_client.c under $(BMTL_UMA) or $(BMTL_UMA_ALT)"; \
	  exit 1; \
	fi
	@test -f "$(UMA_WAN_OPS_C)" || (echo "missing $(UMA_WAN_OPS_C)"; exit 1)

$(BMTL_UMA_RESOLVED)/src/libuma_client.a: $(BMTL_UMA_RESOLVED)/src/uma_client.c \
		$(BMTL_UMA_RESOLVED)/include/uma/client.h
	$(MAKE) -C $(BMTL_UMA_RESOLVED) src/libuma_client.a

$(UMA_WAN_OPS_O): $(UMA_WAN_OPS_C) $(BMTL_UMA_RESOLVED)/src/uma_wan_ops.h
	$(CLANG) $(CFLAGS) -c -o $@ $(UMA_WAN_OPS_C)

wan-cli: $(WAN_OBJS) $(UMA_WAN_OPS_O) $(BMTL_UMA_RESOLVED)/src/libuma_client.a
	$(CLANG) $(CFLAGS) -o $@ $(WAN_OBJS) $(UMA_WAN_OPS_O) $(LDFLAGS)

%.o: %.c include/wan_internal.h
	$(CLANG) $(CFLAGS) -c -o $@ $<

# Caps / gen_* live in wan_ctx; header changes must rebuild all TUs.
$(WAN_OBJS): include/wan_internal.h

tests/%.o: tests/%.c
	$(CLANG) $(CFLAGS) -c -o $@ $<

tests/test_sched_unipc: tests/test_sched_unipc.o sched_unipc.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_sched_unipc.o sched_unipc.o -lm

tests/test_validate: tests/test_validate.o wan_config.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_validate.o wan_config.o

tests/test_gguf_roundtrip: tests/test_gguf_roundtrip.o gguf_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_gguf_roundtrip.o gguf_min.o

tests/test_ops_wire: tests/test_ops_wire.o wan_graph.o wan_weights.o gguf_min.o safetensors_min.o zip_weight.o uma_buf_load.o wan_profile.o $(UMA_WAN_OPS_O)
	$(CLANG) $(CFLAGS) -o $@ tests/test_ops_wire.o wan_graph.o wan_weights.o gguf_min.o safetensors_min.o zip_weight.o uma_buf_load.o wan_profile.o $(UMA_WAN_OPS_O) $(LDFLAGS)

tests/test_tokenizer_spm: tests/test_tokenizer_spm.o tokenizer_spm.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_tokenizer_spm.o tokenizer_spm.o

test: $(TEST_BINS)
	./tests/test_sched_unipc
	./tests/test_validate
	./tests/test_gguf_roundtrip
	./tests/test_ops_wire
	./tests/test_tokenizer_spm

# --- Linux CUDA twin lab (no xcrun / uma_daemon) ---
# Needs NVIDIA driver libs on CT 1564, e.g.:
#   export LD_LIBRARY_PATH=/root/nvidia-host:/usr/lib/ollama/cuda_v13:/usr/local/cuda/lib64
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
DIT_PAGER ?= $(abspath $(CURDIR)/../dit_pager)
LINUX_CFLAGS = -O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -I$(CURDIR)/include -I$(DIT_PAGER)
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -L/usr/lib/ollama/cuda_v13 -lcublas -lcudart -lpthread -lm
# Prefer host NVIDIA libs when present (Proxmox CT).
export LD_LIBRARY_PATH := /root/nvidia-host:/usr/lib/ollama/cuda_v13:$(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)

.PHONY: cuda-smoke cuda-fragment host-fragment cuda-ffn-block cuda-block0 \
	cuda-block0-real cuda-block0-rematch block0-fixture \
	multiblock-fixture cuda-multiblock-rematch \
	latent-fixture cuda-latent-unipc-rematch cuda-lab

backend_cuda.o: backend_cuda.c include/wan_backend.h include/backend_ops.h
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -c -o $@ backend_cuda.c

backend_host.o: backend_host.c include/wan_backend.h include/backend_ops.h
	$(CC) $(LINUX_CFLAGS) -c -o $@ backend_host.c

backend_ops.o: backend_ops.c include/backend_ops.h
	$(CC) $(LINUX_CFLAGS) -c -o $@ backend_ops.c

# RTX 5080 = sm_120; override with CUDA_ARCH=sm_89 etc.
CUDA_ARCH ?= sm_120
backend_cuda_elem.o: backend_cuda_elem.cu include/backend_ops.h
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -I$(CURDIR)/include -c -o $@ backend_cuda_elem.cu

dit_resident.o: dit_resident.c include/dit_resident.h
	$(CC) $(LINUX_CFLAGS) -c -o $@ dit_resident.c

tests/cuda_smoke: tests/cuda_smoke.c backend_cuda.o backend_cuda_elem.o backend_ops.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_smoke.c backend_cuda.o \
	  backend_cuda_elem.o backend_ops.o $(CUDA_LDFLAGS) -lstdc++ -lm

tests/cuda_fragment: tests/cuda_fragment.c backend_cuda.o backend_cuda_elem.o backend_ops.o dit_resident.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_fragment.c backend_cuda.o \
	  backend_cuda_elem.o backend_ops.o dit_resident.o $(DIT_PAGER)/dit_pager.o $(CUDA_LDFLAGS) -lstdc++ -lm

tests/host_fragment: tests/host_fragment.c backend_host.o backend_ops.o dit_resident.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -o $@ tests/host_fragment.c backend_host.o backend_ops.o dit_resident.o \
	  $(DIT_PAGER)/dit_pager.o -lm

tests/cuda_ffn_block: tests/cuda_ffn_block.c backend_cuda.o backend_cuda_elem.o backend_host.o \
	backend_ops.o dit_resident.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_ffn_block.c \
	  backend_cuda.o backend_cuda_elem.o backend_host.o backend_ops.o dit_resident.o \
	  $(DIT_PAGER)/dit_pager.o $(CUDA_LDFLAGS) -lstdc++ -lm

tests/cuda_block0: tests/cuda_block0.c backend_cuda.o backend_cuda_elem.o backend_host.o \
	backend_ops.o dit_resident.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_block0.c \
	  backend_cuda.o backend_cuda_elem.o backend_host.o backend_ops.o dit_resident.o \
	  $(DIT_PAGER)/dit_pager.o $(CUDA_LDFLAGS) -lstdc++ -lm

safetensors_min.o: safetensors_min.c include/safetensors_min.h
	$(CC) $(LINUX_CFLAGS) -c -o $@ safetensors_min.c

tests/cuda_block0_real: tests/cuda_block0_real.c backend_cuda.o backend_cuda_elem.o backend_host.o \
	backend_ops.o dit_resident.o safetensors_min.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_block0_real.c \
	  backend_cuda.o backend_cuda_elem.o backend_host.o backend_ops.o dit_resident.o \
	  safetensors_min.o $(DIT_PAGER)/dit_pager.o $(CUDA_LDFLAGS) -lstdc++ -lm

tests/cuda_block0_rematch: tests/cuda_block0_rematch.c backend_cuda.o backend_cuda_elem.o \
	backend_ops.o safetensors_min.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_block0_rematch.c \
	  backend_cuda.o backend_cuda_elem.o backend_ops.o safetensors_min.o \
	  $(CUDA_LDFLAGS) -lstdc++ -lm

sched_unipc_linux.o: sched_unipc.c include/sched_unipc.h
	$(CC) $(LINUX_CFLAGS) -c -o $@ sched_unipc.c

tests/cuda_multiblock_rematch: tests/cuda_multiblock_rematch.c backend_cuda.o \
	backend_cuda_elem.o backend_ops.o safetensors_min.o dit_resident.o \
	sched_unipc_linux.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_multiblock_rematch.c \
	  backend_cuda.o backend_cuda_elem.o backend_ops.o safetensors_min.o \
	  dit_resident.o sched_unipc_linux.o $(DIT_PAGER)/dit_pager.o \
	  $(CUDA_LDFLAGS) -lstdc++ -lm

tests/cuda_latent_unipc_rematch: tests/cuda_latent_unipc_rematch.c backend_cuda.o \
	backend_cuda_elem.o backend_ops.o safetensors_min.o dit_resident.o \
	sched_unipc_linux.o $(DIT_PAGER)/dit_pager.o
	$(MAKE) -C $(DIT_PAGER) dit_pager.o
	$(CC) $(LINUX_CFLAGS) -I$(CUDA_HOME)/include -o $@ tests/cuda_latent_unipc_rematch.c \
	  backend_cuda.o backend_cuda_elem.o backend_ops.o safetensors_min.o \
	  dit_resident.o sched_unipc_linux.o $(DIT_PAGER)/dit_pager.o \
	  $(CUDA_LDFLAGS) -lstdc++ -lm

cuda-smoke: tests/cuda_smoke
	./tests/cuda_smoke

cuda-fragment: tests/cuda_fragment
	WAN_DIT_RESIDENT=2 ./tests/cuda_fragment

host-fragment: tests/host_fragment
	WAN_DIT_RESIDENT=2 ./tests/host_fragment

cuda-ffn-block: tests/cuda_ffn_block
	WAN_DIT_RESIDENT=2 ./tests/cuda_ffn_block

cuda-block0: tests/cuda_block0
	WAN_DIT_RESIDENT=2 ./tests/cuda_block0

cuda-block0-real: tests/cuda_block0_real
	WAN_DIT_RESIDENT=2 ./tests/cuda_block0_real

block0-fixture:
	WAN_FORCE_SDPA=1 $(or $(WAN_PYTHON),/root/.zerollama/third_party/wan/venv/bin/python) \
	  tools/gen_block0_cuda_fixture.py

multiblock-fixture:
	WAN_FORCE_SDPA=1 $(or $(WAN_PYTHON),/root/.zerollama/third_party/wan/venv/bin/python) \
	  tools/gen_multiblock_cuda_fixture.py

latent-fixture:
	WAN_FORCE_SDPA=1 $(or $(WAN_PYTHON),/root/.zerollama/third_party/wan/venv/bin/python) \
	  tools/gen_latent_unipc_fixture.py

cuda-block0-rematch: tests/cuda_block0_rematch
	WAN_BLOCK0_FIXDIR=$(CURDIR)/dumps/block0_cuda_fixture ./tests/cuda_block0_rematch

cuda-multiblock-rematch: tests/cuda_multiblock_rematch
	WAN_DIT_RESIDENT=2 WAN_UNIPC_ORDER=3 \
	  WAN_MULTIBLOCK_FIXDIR=$(CURDIR)/dumps/multiblock_cuda_fixture \
	  ./tests/cuda_multiblock_rematch

cuda-latent-unipc-rematch: tests/cuda_latent_unipc_rematch
	WAN_DIT_RESIDENT=2 WAN_UNIPC_ORDER=3 \
	  WAN_LATENT_FIXDIR=$(CURDIR)/dumps/latent_unipc_fixture \
	  ./tests/cuda_latent_unipc_rematch

cuda-lab: cuda-smoke cuda-fragment host-fragment cuda-ffn-block cuda-block0 cuda-block0-real

clean:
	rm -f wan-cli $(WAN_OBJS) $(UMA_WAN_OPS_O) tests/*.o $(TEST_BINS) \
	  backend_cuda.o backend_host.o backend_ops.o backend_cuda_elem.o dit_resident.o \
	  safetensors_min.o sched_unipc_linux.o \
	  tests/cuda_smoke tests/cuda_fragment tests/host_fragment tests/cuda_ffn_block \
	  tests/cuda_block0 tests/cuda_block0_real tests/cuda_block0_rematch \
	  tests/cuda_multiblock_rematch tests/cuda_latent_unipc_rematch

