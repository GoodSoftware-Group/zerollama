ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
OUT ?= ane-prefill-bench
METAL_OUT ?= metal-prefill-bench
MPS_OUT ?= metal-prefill-mps-bench
HANDOFF_OUT ?= ane-prefill-handoff-smoke
FFN_SLICE_OUT ?= ane-prefill-ffn-slice-smoke
FFN_POLICY_OUT ?= ane-prefill-ffn-policy-smoke
FFN_FORCE_OUT ?= ane-prefill-ffn-force-smoke
FFN_FORCE_DYLIB ?= libane_ffn_force.dylib
FFN_SWIGLU_FORCE_OUT ?= ane-prefill-ffn-swiglu-force-smoke
FFN_FUSE_UNIT_OUT ?= ane-prefill-ffn-fuse-unit-smoke

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR) -I. -I../../ml/backend/ggml/ggml/src/ggml-metal
GGML_INC := -I../../ml/backend/ggml/ggml/include
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path
METAL_LDFLAGS = -framework Foundation -framework Metal
MPS_LDFLAGS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
HANDOFF_LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -framework Metal -ldl \
	-Wl,-rpath,@loader_path
FFN_SLICE_LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -framework Metal -ldl \
	-Wl,-rpath,@loader_path
FFN_FORCE_LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path
FFN_SWIGLU_FORCE_LDFLAGS = $(FFN_FORCE_LDFLAGS) -framework Metal

POLICY_SRC := ../../ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_policy.c
POLICY_HDR := ../../ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_policy.h
PACK_SRC := ../../ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_force_pack.c
NAMES_SRC := ../../ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_swiglu_names.c
FUSE_SRC := ../../ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_swiglu_fuse.c

.PHONY: all clean bridge

all: bridge $(OUT) $(METAL_OUT) $(MPS_OUT) $(HANDOFF_OUT) $(FFN_SLICE_OUT) $(FFN_POLICY_OUT) $(FFN_FORCE_OUT) $(FFN_FORCE_DYLIB) $(FFN_SWIGLU_FORCE_OUT) $(FFN_FUSE_UNIT_OUT)

bridge:
	$(MAKE) -C $(BRIDGE_DIR)

$(OUT): prefill_bench.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ prefill_bench.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(METAL_OUT): metal_prefill_bench.m
	$(CC) $(CFLAGS) -o $@ metal_prefill_bench.m $(METAL_LDFLAGS)

$(MPS_OUT): metal_prefill_mps_bench.m
	$(CC) $(CFLAGS) -o $@ metal_prefill_mps_bench.m $(MPS_LDFLAGS)

$(HANDOFF_OUT): prefill_handoff_smoke.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ prefill_handoff_smoke.m $(HANDOFF_LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(FFN_SLICE_OUT): prefill_ffn_slice_smoke.m ane_prefill_session.m ane_prefill_session.h ane_ffn_policy.h $(POLICY_SRC) $(POLICY_HDR) $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ prefill_ffn_slice_smoke.m ane_prefill_session.m $(POLICY_SRC) $(FFN_SLICE_LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(FFN_POLICY_OUT): prefill_ffn_policy_smoke.m ane_ffn_policy.h $(POLICY_SRC) $(POLICY_HDR)
	$(CC) $(CFLAGS) -o $@ prefill_ffn_policy_smoke.m $(POLICY_SRC) -framework Foundation

$(FFN_FORCE_OUT): prefill_ffn_force_smoke.m ane_ffn_force_replace.m ane_ffn_force_replace.h \
		ane_prefill_session.m ane_prefill_session.h ane_ffn_policy.h $(POLICY_SRC) $(POLICY_HDR) \
		$(PACK_SRC) $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ prefill_ffn_force_smoke.m ane_ffn_force_replace.m ane_prefill_session.m \
		$(POLICY_SRC) $(PACK_SRC) $(FFN_FORCE_LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(FFN_SWIGLU_FORCE_OUT): prefill_ffn_swiglu_force_smoke.m ane_ffn_force_replace.m ane_ffn_force_replace.h \
		ane_prefill_session.m ane_prefill_session.h ane_ffn_layout_metal.m ane_ffn_layout_metal.h \
		$(POLICY_SRC) $(POLICY_HDR) $(NAMES_SRC) \
		$(PACK_SRC) $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ prefill_ffn_swiglu_force_smoke.m ane_ffn_force_replace.m ane_prefill_session.m \
		ane_ffn_layout_metal.m $(POLICY_SRC) $(NAMES_SRC) $(PACK_SRC) $(FFN_SWIGLU_FORCE_LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(FFN_FORCE_DYLIB): ane_ffn_force_replace.m ane_ffn_force_replace.h ane_prefill_session.m ane_prefill_session.h \
		ane_ffn_layout_metal.m ane_ffn_layout_metal.h $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -dynamiclib -o $@ ane_ffn_force_replace.m ane_prefill_session.m ane_ffn_layout_metal.m \
		$(FFN_SWIGLU_FORCE_LDFLAGS) -install_name @rpath/libane_ffn_force.dylib
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(FFN_FUSE_UNIT_OUT): prefill_ffn_fuse_unit_smoke.c $(FUSE_SRC) $(NAMES_SRC) $(PACK_SRC)
	$(CC) $(CFLAGS) $(GGML_INC) -o $@ prefill_ffn_fuse_unit_smoke.c $(FUSE_SRC) $(NAMES_SRC) $(PACK_SRC) -lm

clean:
	rm -f $(OUT) $(METAL_OUT) $(MPS_OUT) $(HANDOFF_OUT) $(FFN_SLICE_OUT) $(FFN_POLICY_OUT) \
		$(FFN_FORCE_OUT) $(FFN_FORCE_DYLIB) $(FFN_SWIGLU_FORCE_OUT) $(FFN_FUSE_UNIT_OUT)
