ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
OUT ?= ane-prefill-bench
METAL_OUT ?= metal-prefill-bench
MPS_OUT ?= metal-prefill-mps-bench
HANDOFF_OUT ?= ane-prefill-handoff-smoke

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR)
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path
METAL_LDFLAGS = -framework Foundation -framework Metal
MPS_LDFLAGS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
HANDOFF_LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -framework Metal -ldl \
	-Wl,-rpath,@loader_path

.PHONY: all clean bridge

all: bridge $(OUT) $(METAL_OUT) $(MPS_OUT) $(HANDOFF_OUT)

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

clean:
	rm -f $(OUT) $(METAL_OUT) $(MPS_OUT) $(HANDOFF_OUT)
