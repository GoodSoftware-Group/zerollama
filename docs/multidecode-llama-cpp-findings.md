# MultiDecode llama.cpp — findings

**Doc:** [multidecode-llama-cpp.md](./multidecode-llama-cpp.md)

## Lab MD0 (0129, Sep 2026)

| Item | Result |
|------|--------|
| Model | `llama3.2-3b.gguf` (CPU, `-ngl 0`) |
| Forest | prefix 6 + branch_a 1 + branch_b 1 (leaves at same tree depth) |
| `max_abs_diff` leaf_a / leaf_b vs linear | **0** / **0** (`eps=0.02`) |
| Verdict | **PASS** |

## Lab MD1 (0130–0131, Sep 2026)

| Item | Result |
|------|--------|
| Cell `node_id` / `parent_node_id` | Sized with `pos`; threaded through reset/resize/cp/set/rm |
| Mask | Ancestor walk over durable node ids (not ubatch-only `cell_to_tok`) |
| Exactness step2 | Append one token per leaf with `parent_node_id` → logits vs linear **0** / **0** |
| HTTP one-shot | `POST /v1/multidecode` on lab `:18082` → greedy token + `node_ids` |
| HTTP multi-step | `parent_node_ids` + `clear:false` (omit `parent`) → new `node_ids` |
| Go | `capabilities.multidecode`, `/v1/multidecode`, silent Hermes batch LCP pack |

## WHYs

1. **`pos` alone is not enough.** RoPE already accepts custom `llama_batch.pos`. Sibling branches at the same depth get the same RoPE index; standard causal `kv.pos <= q.pos` then leaks across siblings. Ancestor mask is the real patch.

2. **Cell map needs `slot_info` (MD0).** KV cell indices are not guaranteed to equal batch indices after allocation. `set_input_kq_mask_multidecode` uses `sinfos[i_cur]` so cell→token reverse map is exact for the current ubatch.

3. **Multi-step needs durable node ids (MD1).** After the first decode, historical cells are not in the new ubatch. Walking `batch.parent` indices cannot reach them. Cells store `node_id` / `parent_node_id`; the mask walks request parents for new tokens and cell parents for history.

4. **Skip linear continuity checks when `parent` is set.** Tree-depth positions may duplicate and “decrease” across branch packing order; `llama_batch_allocr::init` would reject that under normal rules.

5. **Leave `parent` NULL in `llama_batch_init`.** Allocating an all-`-1` array would look “on” to `parent != NULL` checks and break every normal decode path.

6. **HTTP `parent` optional when `parent_node_ids` set.** Multi-step sticky sessions only need durable parents; synthesizing `parent=[-1,…]` avoids a false “must be an array” 400.

7. **Silent Hermes pack threshold 32.** Short shared prefixes are not worth forest overhead vs Python `generate_batch`; LCP ≥ 32 (or fall through) keeps the wire identical for clients.

## Open

- ctypes / in-process Python forest (later)
- SWA / cross-ubatch forest splits
- Production default GPU smokes (lab CPU `-ngl 0` is the exactness gate)
