# music-c — MiniMax Music 3 host CLI (prompt/chunk/DAV geometry)
#
# Why a new tree (not --family music on video-c): H3 AudioVAE is the wrong hop/rate.
# Why C after mlx: hear a WAV first; Omni CUDA cannot be the Mac first listen.
#
#   make -C x/music-c
#   make -C x/music-c test

CLANG ?= clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -Iinclude
SRCS = main.c music_prompt.c music_chunk.c music_dav.c
OBJS = $(SRCS:.c=.o)

.PHONY: all test clean

all: music-cli

music-cli: $(OBJS)
	$(CLANG) $(CFLAGS) -o $@ $(OBJS) -lm

tests/test_music3_host: tests/test_music3_host.c music_prompt.c music_chunk.c music_dav.c
	$(CLANG) $(CFLAGS) -o $@ tests/test_music3_host.c music_prompt.c music_chunk.c music_dav.c -lm

test: tests/test_music3_host
	./tests/test_music3_host

clean:
	rm -f $(OBJS) music-cli tests/test_music3_host
