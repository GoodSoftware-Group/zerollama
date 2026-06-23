ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
OUT ?= ane-metal-handoff-smoke
DAEMON ?= ane-draft-daemon

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR)
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -framework Metal -ldl \
	-Wl,-rpath,@loader_path

.PHONY: all clean bridge

all: bridge $(OUT) $(DAEMON)

bridge:
	$(MAKE) -C $(BRIDGE_DIR)

$(OUT): metal_handoff_smoke.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ metal_handoff_smoke.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

$(DAEMON): draft_daemon.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ draft_daemon.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

clean:
	rm -f $(OUT) $(DAEMON)
