# Portable DiT pager — Linux/Darwin, no CUDA.
CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -I.

.PHONY: all test clean

all: dit_pager.o

dit_pager.o: dit_pager.c dit_pager.h
	$(CC) $(CFLAGS) -c dit_pager.c -o dit_pager.o

tests/test_lru: tests/test_lru.c dit_pager.o
	$(CC) $(CFLAGS) -o $@ tests/test_lru.c dit_pager.o

test: tests/test_lru
	./tests/test_lru

clean:
	rm -f dit_pager.o tests/test_lru
