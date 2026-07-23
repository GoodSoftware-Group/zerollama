# Build libuma_client + glue for mlxrunner (no embedded uma_sched).
#
# Adopters speak the broker protocol only (PACKAGING.md).
# C sources live in embed/ so cgo does not compile them twice.
#
#   make -C x/mlxrunner/uma
#   CGO_ENABLED=1 go build -tags uma -o zerollama .

BMTL_ROOT ?= $(abspath $(CURDIR)/../../../../bmtl)
BMTL_UMA_TOOLKIT ?= $(BMTL_ROOT)/hardware_lab/lanes/m4/uma_toolkit

SDK   = $(shell xcrun --sdk macosx --show-sdk-path)
CLANG = $(shell xcrun -f clang)

CFLAGS = -O2 -std=c11 -isysroot $(SDK) -I$(BMTL_UMA_TOOLKIT)/include \
	-I$(CURDIR) -fPIC

.PHONY: all clean check-paths

all: libuma_embed.a

check-paths:
	@test -f "$(BMTL_UMA_TOOLKIT)/uma_client.c" || \
	  (echo "missing BMTL_UMA_TOOLKIT=$(BMTL_UMA_TOOLKIT)"; exit 1)

uma_client.o: $(BMTL_UMA_TOOLKIT)/uma_client.c check-paths
	$(CLANG) $(CFLAGS) -c -o $@ $(BMTL_UMA_TOOLKIT)/uma_client.c

uma_glue.o: embed/uma_glue.c uma_glue.h check-paths
	$(CLANG) $(CFLAGS) -c -o $@ embed/uma_glue.c

libuma_embed.a: uma_client.o uma_glue.o
	ar rcs $@ uma_client.o uma_glue.o
	@echo "built $@ (broker client only)"

clean:
	rm -f uma_client.o uma_glue.o libuma_embed.a libuma_sched_core.a
