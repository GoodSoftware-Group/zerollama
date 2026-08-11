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

clean:
	rm -f wan-cli $(WAN_OBJS) $(UMA_WAN_OPS_O) tests/*.o $(TEST_BINS)
