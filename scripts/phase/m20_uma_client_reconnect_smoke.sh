#!/usr/bin/env bash
# Soft reconnect smoke for libuma_client (no daemon TERM).
# Forces half-close of the persistent fd, then PING must succeed again.
#
# Requires live uma_daemon at UMA_SOCK (default /tmp/uma_daemon.sock).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TK="${BMTL_UMA_TOOLKIT:-${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit}"
SOCK="${UMA_SOCK:-/tmp/uma_daemon.sock}"
OUT="${TMPDIR:-/tmp}/m20_uma_reconnect_smoke"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
CLANG="$(xcrun -f clang)"

test -f "${TK}/uma_client.c" || { echo "missing toolkit ${TK}" >&2; exit 1; }
python3 - "${SOCK}" <<'PY'
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2)
s.connect(sys.argv[1])
s.sendall(b"PING\n")
r = s.recv(64)
assert r.startswith(b"OK"), r
print("broker PING ok")
PY

cat > "${OUT}.c" <<'EOF'
#include "uma/client.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Reach into client layout (same as uma_client.c) for soft half-close. */
struct UmaClient {
  char sock[108];
  int fd;
};

int main(void) {
  UmaClient *c = uma_client_connect(NULL);
  if (!c) {
    fprintf(stderr, "FAIL: connect\n");
    return 1;
  }
  if (uma_client_ping(c) != 0) {
    fprintf(stderr, "FAIL: initial ping\n");
    return 1;
  }
  /* Simulate broker drop: close fd, leave handle alive (ensure_fd reconnects). */
  if (c->fd >= 0)
    close(c->fd);
  c->fd = -1;
  if (uma_client_ping(c) != 0) {
    fprintf(stderr, "FAIL: ping after forced half-close (reconnect)\n");
    return 1;
  }
  uma_client_close(c);
  printf("PASS: libuma_client reconnect after half-close\n");
  return 0;
}
EOF

"${CLANG}" -O2 -std=c11 -isysroot "${SDK}" \
  -I"${TK}/include" -o "${OUT}" "${OUT}.c" "${TK}/uma_client.c"
"${OUT}"
rm -f "${OUT}" "${OUT}.c"
