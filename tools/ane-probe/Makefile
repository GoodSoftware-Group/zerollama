# Builds ane-probe against maderix/ane libane_bridge (sibling checkout).
ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
OUT ?= ane-probe
MATMUL_OUT ?= ane-matmul-bench

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR)
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path

.PHONY: all clean bridge install

all: bridge $(OUT) $(MATMUL_OUT)

bridge:
	$(MAKE) -C $(BRIDGE_DIR)

$(OUT): probe.m $(BRIDGE_DIR)/ane_bridge.h $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ probe.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(MATMUL_OUT): matmul_bench.m mil_conv_peak.h $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ matmul_bench.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

clean:
	rm -f $(OUT) $(MATMUL_OUT)

install: all
	@install -d ../../build/ane-probe-darwin/bin
	install -m 755 $(OUT) $(MATMUL_OUT) ../../build/ane-probe-darwin/bin/
