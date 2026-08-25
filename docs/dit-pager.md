# DiT resident pager (`x/dit_pager`)

Portable C11 **N-slot LRU** for DiT block weights. Sits **above** the compute backend (UMA Metal or in-process CUDA).

**Kind:** win + unlock — peak device weights scale with **N**, not `n_layers`; unlocks 16 GB C DiT and later Mac `BANK` eviction without Python `mmgp`.

## API

```c
dit_pager *dit_pager_create(unsigned n_slots);
void dit_pager_destroy(dit_pager *p);
/* Touch layer id; returns slot index to bind. Evicts LRU on miss when full. */
int dit_pager_touch(dit_pager *p, unsigned layer_id, int *evicted_layer_out);
dit_pager_stats dit_pager_get_stats(const dit_pager *p);
```

Default **N=2** (matches antirez `stream_slots[2]` / mmgp-class budgets).

Env (video-c / CUDA lab):

| Variable | Role |
|----------|------|
| `WAN_DIT_RESIDENT` | Max resident DiT blocks (`0` / unset = pager off for product paths). Lab smokes default to **2** when unset. Parsed by `wan_dit_resident_slots()` in `x/video-c/dit_resident.c`. |

## Residency contract (pager ↔ backend)

1. Bank keys are **layer-scoped** (`W_L{id}`), not slot-scoped.
2. On pager miss with eviction: `bank_evict(W_L{evicted})` **before** `bank_put(W_L{new})`.
3. `bank_bind(W_L{id}, "W")` then GEMM — aliases must not double-count device bytes.
4. Kill if backend peak bytes exceed `N * layer_bytes + activations`.

## Kill / success metrics

- Unit tests: deterministic LRU order, hit/miss counts.
- CUDA / host fragment smokes: pager peak **and** backend peak with `N=2` **≪** load-all.
- If fragment cannot show N-scaling residency, stop and rethink before full DiT step investment.

## Related

- [cuda-uma-toolkit.md](./cuda-uma-toolkit.md) — backends under the pager
- [wangp-borrowings.md](./wangp-borrowings.md) — Python mmgp is the bridge; this is the C home
- [video-c.md](./video-c.md) — Pure-C multi-family client (was wan-c)
