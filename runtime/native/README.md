# Native runtime extensions (Phase 15+)

**Why:** PagedAttention block allocation and (later) decode scheduling run on every admission tick. Moving hot bookkeeping off the Python interpreter reduces GIL contention when training and inference share one embedded CPython.

## Shipped (v0)

| Module | Source | Role |
|--------|--------|------|
| `runtime.kv._kv_native` | `native/kv_block_pool.c` | `BlockPool`; `scheduler_tick()`; `decode_step(n)`; `kv_stats()` read-only counters |

## Build

```bash
cd runtime
python3 setup.py build_ext --inplace
PYTHONPATH=. python3 -c "from runtime.kv._kv_native import BlockPool; print(BlockPool(8, 16).num_free)"
```

Enable at runtime:

```bash
export ZEROLLAMA_RUNTIME_KV_NATIVE=1
```

Default remains pure Python (`kv.backend: python` on `/health`). If env is set but the `.so` is missing, logs warn once and health includes a `note` with the build command.

## Tests

```bash
cd runtime && PYTHONPATH=. python -m pytest tests/test_kv_native_parity.py -q
```

CI: optional `pip install -e ".[native]"` before pytest; tests skip if extension not built.

## Next (Phase 15 v1+)

- Wire block ids into in-process llama KV (ctypes) instead of per-request `llama_context`
- Native decode tick / batch builder (C/Rust) with Python config-only layer

See [docs/phase15-native-kv.md](../../docs/phase15-native-kv.md).
