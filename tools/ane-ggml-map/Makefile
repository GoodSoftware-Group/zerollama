OUT ?= ane-ggml-map-smoke

CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -I../ane-metal
LDFLAGS = -framework Foundation -framework IOSurface -framework Metal

.PHONY: all clean

all: $(OUT)

$(OUT): ggml_map_smoke.m
	$(CC) $(CFLAGS) -o $@ ggml_map_smoke.m $(LDFLAGS)

clean:
	rm -f $(OUT)
