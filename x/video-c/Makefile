# video-c — Pure-C multi-family video client (Wan + H3 stub; Darwin UMA / CUDA twin)
#
#   make -C x/video-c test              # weightless rematch
#   make -C x/video-c test-h3-weights   # AudioVAE + video VAE encode/decode if pack present
#   ./video-cli --family h3 --info -d ~/.zerollama/models/MiniMax-H3
#   ./video-cli --family h3 --decode-audio -d DIR -o out.wav
#
# wan-cli is a symlink to video-cli (compat). x/wan-c → x/video-c symlink.
# Links libuma_client from bmtl uma_toolkit (broker protocol only).
# Host kernels: compiles uma_wan_ops.c from the toolkit tree.
# Darwin (this track) vs CUDA lab targets (other owner) — see docs/video-c.md.

BMTL_ROOT ?= $(abspath $(CURDIR)/../../../bmtl)
BMTL_UMA ?= $(BMTL_ROOT)/targets/m4-max/hardware/uma_toolkit
BMTL_UMA_ALT ?= $(BMTL_ROOT)/hardware_lab/lanes/m4/uma_toolkit

SDK   = $(shell xcrun --sdk macosx --show-sdk-path)
CLANG = $(shell xcrun -f clang)

WAN_SRCS = \
	main.c \
	h3_info.c \
	family_h3/h3_host.c \
	family_h3/h3_reuse.c \
	family_h3/h3_st_store.c \
	family_h3/h3_audio_vae_host.c \
	family_h3/h3_audio_vae_decode.c \
	family_h3/h3_adaln_host.c \
	family_h3/h3_clipproj_host.c \
	family_h3/h3_dit_host.c \
	family_h3/h3_dit_block.c \
	family_h3/h3_dit_forward.c \
	family_h3/h3_dit_pack.c \
	family_h3/h3_convrot.c \
	family_h3/h3_video_vae_host.c \
	family_h3/h3_video_vae_encode.c \
	family_h3/h3_video_vae_decode.c \
	family_h3/h3_tokenizer.c \
	family_h3/h3_qwen_te_host.c \
	family_h3/h3_present.c \
	family_h3/h3_text_cond.c \
	family_h3/h3_qwen_te_4b.c \
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

TEST_SRCS = tests/test_sched_unipc.c tests/test_validate.c tests/test_gguf_roundtrip.c tests/test_ops_wire.c tests/test_tokenizer_spm.c tests/test_h3_info.c tests/test_h3_host.c tests/test_h3_st_store.c tests/test_h3_packing_mlx.c tests/test_h3_audio_vae_host.c tests/test_h3_adaln_host.c tests/test_h3_clipproj_host.c tests/test_h3_dit_host.c tests/test_h3_video_vae_host.c tests/test_encode_avi.c tests/test_encode_mp4.c tests/test_h3_tokenizer.c tests/test_h3_qwen_te_host.c tests/test_h3_present.c tests/test_h3_convrot.c tests/test_h3_dit_block.c tests/test_h3_dit_forward.c tests/test_h3_dit_pack.c tests/test_h3_text_cond.c
TEST_BINS = tests/test_sched_unipc tests/test_validate tests/test_gguf_roundtrip tests/test_ops_wire tests/test_tokenizer_spm tests/test_h3_info tests/test_h3_host tests/test_h3_st_store tests/test_h3_packing_mlx tests/test_h3_audio_vae_host tests/test_h3_adaln_host tests/test_h3_clipproj_host tests/test_h3_dit_host tests/test_h3_video_vae_host tests/test_encode_avi tests/test_encode_mp4 tests/test_h3_tokenizer tests/test_h3_qwen_te_host tests/test_h3_present tests/test_h3_convrot tests/test_h3_dit_block tests/test_h3_dit_forward tests/test_h3_dit_pack tests/test_h3_text_cond
WEIGHT_TEST_BINS = tests/test_h3_audio_vae_decode tests/test_h3_video_vae_encode tests/test_h3_video_vae_decode tests/test_h3_qwen_te_4b tests/test_h3_convrot tests/test_h3_dit_block tests/test_h3_dit_forward

# Resolve toolkit path (prefer primary).
BMTL_UMA_RESOLVED := $(if $(wildcard $(BMTL_UMA)/src/uma_client.c),$(BMTL_UMA),$(BMTL_UMA_ALT))
UMA_WAN_OPS_C := $(BMTL_UMA_RESOLVED)/src/uma_wan_ops.c
UMA_WAN_OPS_O := uma_wan_ops.o

BMTL_INC = -I$(BMTL_ROOT)/include -I$(BMTL_ROOT)/src/pretokenize
# Gigatoken-derived BMTL tokenizer (no uma_tokenize_shared / lab hello merges).
BMTL_TOK_OBJS = bmtl_tok/bmtl_tokenizer.o bmtl_tok/bmtl_status.o \
	bmtl_tok/bmtl_swar_predicates.o bmtl_tok/bmtl_mask_scanner.o \
	bmtl_tok/bmtl_batch_walker.o bmtl_tok/bmtl_unicode_table.o \
	bmtl_tok/bmtl_scheme_gpt2.o bmtl_tok/bmtl_scheme_cl100k.o \
	bmtl_tok/bmtl_scheme_qwen2.o bmtl_tok/bmtl_dispatch.o \
	bmtl_tok/bmtl_pair_rank_table.o bmtl_tok/bmtl_merge.o \
	bmtl_tok/bmtl_pretoken_cache.o bmtl_tok/bmtl_advance_scalar.o \
	bmtl_tok/bmtl_safe_split.o bmtl_tok/bmtl_nfc.o

CFLAGS = -O2 -std=c11 -Wall -Wextra -isysroot $(SDK) \
	-DACCELERATE_NEW_LAPACK \
	-I$(CURDIR)/include -I$(BMTL_UMA_RESOLVED)/include -I$(BMTL_UMA_RESOLVED)/src \
	$(BMTL_INC) -fPIC

LDFLAGS = -L$(BMTL_UMA_RESOLVED)/src -luma_client -lm -framework Accelerate \
	-framework CoreFoundation

.PHONY: all clean test test-h3-weights check-paths

all: check-paths video-cli

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

video-cli: $(WAN_OBJS) $(UMA_WAN_OPS_O) $(BMTL_TOK_OBJS) $(BMTL_UMA_RESOLVED)/src/libuma_client.a
	$(CLANG) $(CFLAGS) -o $@ $(WAN_OBJS) $(UMA_WAN_OPS_O) $(BMTL_TOK_OBJS) $(LDFLAGS)
	ln -sfn video-cli wan-cli

%.o: %.c include/wan_internal.h
	@mkdir -p $(dir $@)
	$(CLANG) $(CFLAGS) -c -o $@ $<

family_h3/%.o: family_h3/%.c include/h3_host.h include/h3_st_store.h include/h3_audio_vae_host.h include/h3_audio_vae_decode.h include/h3_adaln_host.h include/h3_clipproj_host.h include/h3_dit_host.h include/h3_dit_block.h include/h3_dit_forward.h include/h3_dit_pack.h include/h3_text_cond.h include/h3_convrot.h include/h3_video_vae_host.h include/h3_video_vae_encode.h include/h3_video_vae_decode.h
	@mkdir -p $(dir $@)
	$(CLANG) $(CFLAGS) -c -o $@ $<

tests/test_h3_audio_vae_decode: tests/test_h3_audio_vae_decode.o family_h3/h3_audio_vae_decode.o family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_audio_vae_decode.o family_h3/h3_audio_vae_decode.o \
	  family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o \
	  -framework Accelerate -lm

h3_info.o: h3_info.c include/h3_info.h include/h3_host.h include/h3_adaln_host.h include/h3_dit_host.h include/h3_clipproj_host.h include/h3_audio_vae_host.h include/h3_st_store.h include/h3_video_vae_host.h
	$(CLANG) $(CFLAGS) -c -o $@ h3_info.c

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

tests/test_h3_info: tests/test_h3_info.o h3_info.o family_h3/h3_host.o family_h3/h3_reuse.o family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_audio_vae_host.o family_h3/h3_adaln_host.o family_h3/h3_dit_host.o family_h3/h3_clipproj_host.o family_h3/h3_video_vae_host.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_info.o h3_info.o family_h3/h3_host.o family_h3/h3_reuse.o \
	  family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_audio_vae_host.o family_h3/h3_adaln_host.o \
	  family_h3/h3_dit_host.o family_h3/h3_clipproj_host.o family_h3/h3_video_vae_host.o \
	  safetensors_min.o -framework Accelerate -lm

tests/test_h3_host: tests/test_h3_host.o family_h3/h3_host.o family_h3/h3_reuse.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_host.o family_h3/h3_host.o family_h3/h3_reuse.o \
	  -framework Accelerate -lm

tests/test_h3_st_store: tests/test_h3_st_store.o family_h3/h3_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_st_store.o family_h3/h3_host.o family_h3/h3_st_store.o \
	  family_h3/h3_convrot.o safetensors_min.o -framework Accelerate -lm

tests/test_h3_packing_mlx: tests/test_h3_packing_mlx.o family_h3/h3_host.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_packing_mlx.o family_h3/h3_host.o \
	  -framework Accelerate -lm

tests/test_h3_audio_vae_host: tests/test_h3_audio_vae_host.o family_h3/h3_audio_vae_host.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_audio_vae_host.o family_h3/h3_audio_vae_host.o -framework Accelerate -lm

tests/test_h3_adaln_host: tests/test_h3_adaln_host.o family_h3/h3_adaln_host.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_adaln_host.o family_h3/h3_adaln_host.o -lm

tests/test_h3_clipproj_host: tests/test_h3_clipproj_host.o family_h3/h3_clipproj_host.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_clipproj_host.o family_h3/h3_clipproj_host.o \
	  safetensors_min.o -framework Accelerate -lm

tests/test_h3_dit_host: tests/test_h3_dit_host.o family_h3/h3_dit_host.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_dit_host.o family_h3/h3_dit_host.o \
	  -framework Accelerate -lm

tests/test_h3_video_vae_host: tests/test_h3_video_vae_host.o family_h3/h3_video_vae_host.o family_h3/h3_dit_host.o family_h3/h3_host.o family_h3/h3_reuse.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_video_vae_host.o family_h3/h3_video_vae_host.o \
	  family_h3/h3_dit_host.o family_h3/h3_host.o family_h3/h3_reuse.o -framework Accelerate -lm

tests/test_encode_avi: tests/test_encode_avi.o encode_mp4.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_encode_avi.o encode_mp4.o

tests/test_encode_mp4: tests/test_encode_mp4.o encode_mp4.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_encode_mp4.o encode_mp4.o

tests/test_h3_tokenizer: tests/test_h3_tokenizer.o family_h3/h3_tokenizer.o $(BMTL_TOK_OBJS)
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_tokenizer.o family_h3/h3_tokenizer.o \
	  $(BMTL_TOK_OBJS) -framework CoreFoundation

tests/test_h3_qwen_te_host: tests/test_h3_qwen_te_host.o family_h3/h3_qwen_te_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_host.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_qwen_te_host.o family_h3/h3_qwen_te_host.o \
	  family_h3/h3_dit_host.o family_h3/h3_audio_vae_host.o -framework Accelerate -lm

tests/test_h3_present: tests/test_h3_present.o family_h3/h3_present.o family_h3/h3_tokenizer.o $(BMTL_TOK_OBJS)
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_present.o family_h3/h3_present.o \
	  family_h3/h3_tokenizer.o $(BMTL_TOK_OBJS) -framework CoreFoundation

tests/test_h3_convrot: tests/test_h3_convrot.o family_h3/h3_convrot.o family_h3/h3_st_store.o family_h3/h3_dit_host.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_convrot.o family_h3/h3_convrot.o \
	  family_h3/h3_st_store.o family_h3/h3_dit_host.o safetensors_min.o \
	  -framework Accelerate -lm

tests/test_h3_dit_block: tests/test_h3_dit_block.o family_h3/h3_dit_block.o family_h3/h3_dit_host.o family_h3/h3_adaln_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_dit_block.o family_h3/h3_dit_block.o \
	  family_h3/h3_dit_host.o family_h3/h3_adaln_host.o family_h3/h3_st_store.o \
	  family_h3/h3_convrot.o safetensors_min.o -framework Accelerate -lm

tests/test_h3_dit_forward: tests/test_h3_dit_forward.o family_h3/h3_dit_pack.o family_h3/h3_dit_forward.o family_h3/h3_dit_block.o family_h3/h3_dit_host.o family_h3/h3_adaln_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_host.o family_h3/h3_reuse.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_dit_forward.o family_h3/h3_dit_pack.o \
	  family_h3/h3_dit_forward.o family_h3/h3_dit_block.o family_h3/h3_dit_host.o \
	  family_h3/h3_adaln_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o \
	  family_h3/h3_host.o family_h3/h3_reuse.o safetensors_min.o \
	  -framework Accelerate -lm

tests/test_h3_dit_pack: tests/test_h3_dit_pack.o family_h3/h3_dit_pack.o family_h3/h3_dit_forward.o family_h3/h3_dit_block.o family_h3/h3_dit_host.o family_h3/h3_adaln_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_host.o family_h3/h3_reuse.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_dit_pack.o family_h3/h3_dit_pack.o \
	  family_h3/h3_dit_forward.o family_h3/h3_dit_block.o family_h3/h3_dit_host.o \
	  family_h3/h3_adaln_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o \
	  family_h3/h3_host.o family_h3/h3_reuse.o safetensors_min.o \
	  -framework Accelerate -lm

tests/test_h3_text_cond: tests/test_h3_text_cond.o family_h3/h3_text_cond.o family_h3/h3_present.o family_h3/h3_tokenizer.o family_h3/h3_clipproj_host.o family_h3/h3_qwen_te_4b.o family_h3/h3_qwen_te_host.o family_h3/h3_dit_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_audio_vae_host.o family_h3/h3_host.o safetensors_min.o $(BMTL_TOK_OBJS)
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_text_cond.o family_h3/h3_text_cond.o \
	  family_h3/h3_present.o family_h3/h3_tokenizer.o family_h3/h3_clipproj_host.o \
	  family_h3/h3_qwen_te_4b.o family_h3/h3_qwen_te_host.o family_h3/h3_dit_host.o \
	  family_h3/h3_st_store.o family_h3/h3_convrot.o family_h3/h3_audio_vae_host.o \
	  family_h3/h3_host.o \
	  safetensors_min.o $(BMTL_TOK_OBJS) -framework Accelerate -framework CoreFoundation -lm

tests/test_h3_qwen_te_4b: tests/test_h3_qwen_te_4b.o family_h3/h3_qwen_te_4b.o family_h3/h3_qwen_te_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_qwen_te_4b.o family_h3/h3_qwen_te_4b.o \
	  family_h3/h3_qwen_te_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_host.o \
	  family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o -framework Accelerate -lm

bmtl_tok/bmtl_tokenizer.o: $(BMTL_ROOT)/src/tokenizer.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_status.o: $(BMTL_ROOT)/src/status.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_nfc.o: $(BMTL_ROOT)/src/pretokenize/nfc.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_swar_predicates.o: $(BMTL_ROOT)/src/pretokenize/swar_predicates.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_mask_scanner.o: $(BMTL_ROOT)/src/pretokenize/mask_scanner.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_batch_walker.o: $(BMTL_ROOT)/src/pretokenize/batch_walker.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_advance_scalar.o: $(BMTL_ROOT)/src/pretokenize/advance_scalar.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_safe_split.o: $(BMTL_ROOT)/src/pretokenize/safe_split.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_unicode_table.o: $(BMTL_ROOT)/src/pretokenize/unicode_table.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_scheme_gpt2.o: $(BMTL_ROOT)/src/pretokenize/scheme_gpt2.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_scheme_cl100k.o: $(BMTL_ROOT)/src/pretokenize/scheme_cl100k.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_scheme_qwen2.o: $(BMTL_ROOT)/src/pretokenize/scheme_qwen2.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_dispatch.o: $(BMTL_ROOT)/src/pretokenize/dispatch.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_pair_rank_table.o: $(BMTL_ROOT)/src/bpe/pair_rank_table.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_merge.o: $(BMTL_ROOT)/src/bpe/merge.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<
bmtl_tok/bmtl_pretoken_cache.o: $(BMTL_ROOT)/src/bpe/pretoken_cache.c
	@mkdir -p bmtl_tok
	$(CLANG) $(CFLAGS) $(BMTL_INC) -c -o $@ $<

tests/test_h3_video_vae_encode: tests/test_h3_video_vae_encode.o family_h3/h3_video_vae_encode.o family_h3/h3_video_vae_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_decode.o family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_video_vae_encode.o family_h3/h3_video_vae_encode.o \
	  family_h3/h3_video_vae_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_decode.o \
	  family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o \
	  -framework Accelerate -lm

tests/test_h3_video_vae_decode: tests/test_h3_video_vae_decode.o family_h3/h3_video_vae_decode.o family_h3/h3_video_vae_encode.o family_h3/h3_video_vae_host.o family_h3/h3_dit_host.o family_h3/h3_audio_vae_decode.o family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o family_h3/h3_convrot.o safetensors_min.o
	$(CLANG) $(CFLAGS) -o $@ tests/test_h3_video_vae_decode.o family_h3/h3_video_vae_decode.o \
	  family_h3/h3_video_vae_encode.o family_h3/h3_video_vae_host.o family_h3/h3_dit_host.o \
	  family_h3/h3_audio_vae_decode.o family_h3/h3_audio_vae_host.o family_h3/h3_st_store.o \
	  family_h3/h3_convrot.o safetensors_min.o -framework Accelerate -lm

test: $(TEST_BINS)
	./tests/test_sched_unipc
	./tests/test_validate
	./tests/test_gguf_roundtrip
	./tests/test_ops_wire
	./tests/test_tokenizer_spm
	./tests/test_h3_info
	./tests/test_h3_host
	./tests/test_h3_st_store
	./tests/test_h3_packing_mlx
	./tests/test_h3_audio_vae_host
	./tests/test_h3_adaln_host
	./tests/test_h3_clipproj_host
	./tests/test_h3_dit_host
	./tests/test_h3_video_vae_host
	./tests/test_encode_avi
	./tests/test_encode_mp4
	./tests/test_h3_tokenizer
	./tests/test_h3_qwen_te_host
	./tests/test_h3_present
	./tests/test_h3_convrot
	./tests/test_h3_dit_block
	./tests/test_h3_dit_forward
	./tests/test_h3_dit_pack
	./tests/test_h3_text_cond

# Real MiniMax-H3 weights. Skips if pack absent.
# AudioVAE ~7s; video encode 32²; ViT decode T=2/T=7 + roundtrip.
test-h3-weights: tests/test_h3_audio_vae_decode tests/test_h3_video_vae_encode tests/test_h3_video_vae_decode tests/test_h3_qwen_te_4b tests/test_h3_convrot tests/test_h3_dit_block tests/test_h3_dit_forward
	./tests/test_h3_audio_vae_decode
	./tests/test_h3_video_vae_encode
	./tests/test_h3_video_vae_decode
	./tests/test_h3_qwen_te_4b
	H3_CONVROT_PACK=1 ./tests/test_h3_convrot
	H3_CONVROT_PACK=1 ./tests/test_h3_dit_block
	H3_CONVROT_PACK=1 ./tests/test_h3_dit_forward

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
	rm -f video-cli wan-cli $(WAN_OBJS) $(UMA_WAN_OPS_O) tests/*.o $(TEST_BINS) \
	  $(WEIGHT_TEST_BINS) $(BMTL_TOK_OBJS) \
	  family_h3/*.o \
	  backend_cuda.o backend_host.o backend_ops.o backend_cuda_elem.o dit_resident.o \
	  safetensors_min.o sched_unipc_linux.o \
	  tests/cuda_smoke tests/cuda_fragment tests/host_fragment tests/cuda_ffn_block \
	  tests/cuda_block0 tests/cuda_block0_real tests/cuda_block0_rematch \
	  tests/cuda_multiblock_rematch tests/cuda_latent_unipc_rematch

