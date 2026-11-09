# ANE draft hook patches (zerollama lab track)

Canonical ANE hook **sources** live in `canonical/common/` so vendor rsync cannot delete them.

| Artifact | Role |
|----------|------|
| `canonical/common/ane_draft_*` | Source of truth copied to sibling + in-tree |
| `apply_speculative_ane_hook.py` | Idempotent speculative.cpp + CMakeLists wiring (B1–B7) |
| `apply_iosurface_sibling.py` | ggml Metal IOSurface export API (`strncmp` on `MTL*` device names) |
| `regenerate_0018_patch.sh` | Commit hook on `vendor/llama-cpp-*` → `llama/patches/0018-*.patch` |

## Operator flow

```bash
# Automatic on Darwin build (after pin checkout):
./scripts/build/build_llama_server.sh

# Manual sync (defaults to vendor/llama-cpp-<pin>, falls back to ../llama.cpp):
./scripts/vendor/sync_ane_hook_to_llama_cpp.sh

# After vendor rebase (--sync restores in-tree ANE):
./scripts/vendor/rebase_vendor_unified.sh --sync

# Promote to formal patch series (once per hook milestone):
./tools/ane-patches/regenerate_0018_patch.sh
make -f Makefile.sync clean apply-patches
```

Skip auto-sync: `ZEROLLAMA_SKIP_ANE_HOOK_SYNC=1 ./scripts/build/build_llama_server.sh`

## Doctor

`zerollama doctor` includes **ane draft hook (llama.cpp)** — checks sibling sources + built binary for B7 markers.

`zerollama doctor --fix` runs `sync_ane_hook` before `build_llama_server` when sources are stale.
