ANE_REPO ?= $(HOME)/Sites/inference/ane
BRIDGE_DIR := $(ANE_REPO)/bridge
OUT ?= ane-iosurface-smoke

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I$(BRIDGE_DIR)
LDFLAGS = -L$(BRIDGE_DIR) -lane_bridge -framework Foundation -framework IOSurface -ldl \
	-Wl,-rpath,@loader_path

.PHONY: all clean bridge

all: bridge $(OUT)

bridge:
	$(MAKE) -C $(BRIDGE_DIR)

$(OUT): iosurface_smoke.m $(BRIDGE_DIR)/libane_bridge.dylib
	$(CC) $(CFLAGS) -o $@ iosurface_smoke.m $(LDFLAGS)
	install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib $@

clean:
	rm -f $(OUT)
