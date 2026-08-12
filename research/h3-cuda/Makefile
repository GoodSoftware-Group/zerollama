# Lab-only MiniMax-H3 CUDA backend. Do not bind production ports.
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -std=c++17 -O2 -arch=$(ARCH) -Iinclude --expt-relaxed-constexpr
LDFLAGS ?= -lcublas -L$(CUDA_HOME)/lib64

OBJS := \
  bin/h3_cuda_runtime.o \
  bin/h3_cuda_tensor.o \
  bin/h3_cuda_gemm.o \
  bin/h3_cuda_attn.o \
  bin/h3_cuda_dit_ops.o \
  bin/h3_cuda_p1_text.o \
  bin/h3_cuda_p2_elem.o \
  bin/h3_cuda_token.o \
  bin/h3_cuda_conv.o \
  bin/h3_cuda_audio.o \
  bin/h3_cuda_vae.o \
  bin/h3_cuda_f32_dit.o \
  bin/h3_cuda_int8.o \
  bin/h3_cuda_stubs.o

.PHONY: all smoke p0 dit p1 pack p2 token audio vae f32dit int8 dit8 bench8 fixture link-dit audio-vae clean

all: p0 p2 token audio vae f32dit int8 dit8 fixture link-dit

bin/%.o: src/cuda/%.cu include/h3_gpu.h include/h3_cuda_internal.h
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/libh3_cuda.a: $(OBJS)
	ar rcs $@ $^

bin/test_cuda_probe: tests/test_cuda_probe.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_p0_ops: tests/test_p0_ops.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_synthetic_dit_block: tests/test_synthetic_dit_block.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_p1_text: tests/test_p1_text.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_dit_pack: tests/test_dit_pack.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_p2_elem: tests/test_p2_elem.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_token_conv: tests/test_token_conv.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_audio_vision: tests/test_audio_vision.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_vae: tests/test_vae.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_f32_dit: tests/test_f32_dit.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_int8: tests/test_int8.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/test_synthetic_dit_int8: tests/test_synthetic_dit_int8.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

bin/bench_linear_int8: tests/bench_linear_int8.cu bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< bin/libh3_cuda.a $(LDFLAGS)

PORTABLE_H3 ?= ../h3.c
# Compile safetensors as C (nvcc would C++-mangle decls without extern "C" in the header).
bin/portable/h3_safetensors.o: $(PORTABLE_H3)/h3_safetensors.c
	mkdir -p bin/portable
	$(CC) $(PORTABLE_CFLAGS) -c $< -o $@

bin/test_toy_fixture: tests/test_toy_fixture.cu bin/libh3_cuda.a bin/portable/h3_safetensors.o
	$(NVCC) $(NVCCFLAGS) -I$(PORTABLE_H3) -o $@ tests/test_toy_fixture.cu \
		bin/portable/h3_safetensors.o bin/libh3_cuda.a $(LDFLAGS)

smoke: bin/test_cuda_probe bin/test_p0_ops
	./bin/test_cuda_probe
	./bin/test_p0_ops

dit: bin/test_synthetic_dit_block
	./bin/test_synthetic_dit_block

p1: bin/test_p1_text
	./bin/test_p1_text

pack: bin/test_dit_pack
	./bin/test_dit_pack

p2: bin/test_p2_elem
	./bin/test_p2_elem

token: bin/test_token_conv
	./bin/test_token_conv

audio: bin/test_audio_vision
	./bin/test_audio_vision

vae: bin/test_vae
	./bin/test_vae

f32dit: bin/test_f32_dit
	./bin/test_f32_dit

int8: bin/test_int8
	./bin/test_int8

dit8: bin/test_synthetic_dit_int8
	./bin/test_synthetic_dit_int8

bench8: bin/bench_linear_int8
	./bin/bench_linear_int8

FIXTURE ?= misc/fixtures/h3_dit_toy_f32.safetensors
fixture: bin/test_toy_fixture $(FIXTURE)
	./bin/test_toy_fixture $(FIXTURE)

$(FIXTURE): scripts/generate_toy_dit_fixture.py
	python3 scripts/generate_toy_dit_fixture.py -o $@

p0: smoke dit p1 pack

# Link portable antirez DiT C against CUDA backend (compile/link only).
PORTABLE_H3 ?= ../h3.c
PORTABLE_CFLAGS := -std=c11 -O2 -D_POSIX_C_SOURCE=200809L -Iinclude -I$(PORTABLE_H3)
bin/libh3_dit_portable.a: $(PORTABLE_H3)/h3_dit.c $(PORTABLE_H3)/h3_dit_schedule.c \
		$(PORTABLE_H3)/h3_weights.c $(PORTABLE_H3)/h3_safetensors.c \
		src/host/h3_host_linux_stub.c include/h3_gpu.h
	mkdir -p bin/portable
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_dit.c -o bin/portable/h3_dit.o
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_dit_schedule.c -o bin/portable/h3_dit_schedule.o
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_weights.c -o bin/portable/h3_weights.o
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_safetensors.c -o bin/portable/h3_safetensors.o
	$(CC) $(PORTABLE_CFLAGS) -c src/host/h3_host_linux_stub.c -o bin/portable/h3_host_linux_stub.o
	ar rcs $@ bin/portable/*.o

bin/test_dit_link: tests/test_dit_link.c bin/libh3_dit_portable.a bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -I$(PORTABLE_H3) -o $@ $< bin/libh3_dit_portable.a bin/libh3_cuda.a $(LDFLAGS) -lpthread -lm

link-dit: bin/test_dit_link
	./bin/test_dit_link

# Portable antirez AudioVAE + CUDA backend against real FL2VA weights.
AUDIO_VAE_DIR ?= /tmp/h3c-research/weights/FL2VA/audio_vae
AUDIO_VAE_T ?= 4
bin/libh3_audio_portable.a: $(PORTABLE_H3)/h3_audio_vae.c \
		$(PORTABLE_H3)/h3_weights.c $(PORTABLE_H3)/h3_safetensors.c \
		include/h3_gpu.h
	mkdir -p bin/portable
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_audio_vae.c -o bin/portable/h3_audio_vae.o
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_weights.c -o bin/portable/h3_weights.o
	$(CC) $(PORTABLE_CFLAGS) -c $(PORTABLE_H3)/h3_safetensors.c -o bin/portable/h3_safetensors.o
	ar rcs $@ bin/portable/h3_audio_vae.o bin/portable/h3_weights.o \
	  bin/portable/h3_safetensors.o

bin/test_audio_vae_decode: tests/test_audio_vae_decode.c bin/libh3_audio_portable.a \
		bin/libh3_cuda.a
	$(NVCC) $(NVCCFLAGS) -I$(PORTABLE_H3) -o $@ $< bin/libh3_audio_portable.a \
	  bin/libh3_cuda.a $(LDFLAGS) -lpthread -lm

audio-vae: bin/test_audio_vae_decode
	./bin/test_audio_vae_decode $(AUDIO_VAE_DIR) $(AUDIO_VAE_T)

clean:
	rm -rf bin
