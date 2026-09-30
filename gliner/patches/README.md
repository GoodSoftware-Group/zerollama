# GLiNER.cpp patches (zerollama)

Applied by `scripts/vendor/ensure_gliner_cpp.sh` after checkout of `GLINER_CPP_COMMIT`.

| Patch | WHY |
|-------|-----|
| `0001-token-logits-channel-first-transpose.patch` | onnx-community token ONNX emits `[3,B,L,C]`; upstream TokenDecoder reads `[B,L,C,3]` (Finding 10) |
