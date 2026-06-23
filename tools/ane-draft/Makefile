# ANE tools — probe smoke + peak bench (maderix/ane libane_bridge).
ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
PROBE_DIR := ../ane-probe
CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR) -I$(PROBE_DIR)
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path

.PHONY: all clean bridge

all: bridge ane-draft-bench

bridge:
	$(MAKE) -C $(BRIDGE_DIR)

ane-draft-bench: draft_bench.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ draft_bench.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

clean:
	rm -f ane-draft-bench
