#!/usr/bin/env bash
# Deprecated name — use sync_vendor_llama.sh (pin comes from Makefile.sync FETCH_HEAD).
exec "$(dirname "$0")/sync_vendor_llama.sh" "$@"
