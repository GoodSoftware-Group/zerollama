# Weight-edit interface

Decode → bind → apply for stored weight edits. Patch file formats are codecs.
The checkpoint layout is a store. The `Edit` record is the contract between them.

## Pipeline

```
patch bytes ──Codec.Decode──► Edit ──Bind──► BoundEdit ──Apply──► new directory + tag
                                 ▲
                         Store (source_dir)
```

| Type | Role |
|------|------|
| **Codec** | Pure decoder. `Detect` + `Decode` → `Edit`. Does not open the checkpoint. |
| **Edit** | Name, id, identity claims, and `Change`s (tensor name, dtype, logical shape, row-major indices, old/new stored bytes). |
| **Bind** | Opens the store. Checks every identity claim the edit includes and the old (or new) bytes at the indices. |
| **Store** | Checkpoint. v1 is a Hugging Face directory (`config.json` + `*.safetensors`) from `ConfigV2.SourceDir`. |
| **Apply** | Writes only a bound edit into a new directory. Restore swaps old/new and checks the result fingerprint. |

Package: [`x/weightedit`](../../../x/weightedit). Codec registration is a package-level list; unknown bytes are HTTP **415**.

## Why `source_dir` only (v1)

A geodesic-patch / DAWE file fingerprints the exact safetensors directory it was authored against. A normal `create` import renames, splits, casts, and quantizes tensors ([`qwen35ImportTransform`](../../../x/create/qwen35.go)), so the hash fails. GGUF is a different encoding. v1 returns **400** when `source_dir` is empty. Linked tags from `zerollama create --experimental --link` are the supported input.

## Codec: geodesic-patch v1

Matches [bandr-ai/dawe](https://github.com/bandr-ai/dawe) `dawe/patch.py`.

- Safetensors file with `__metadata__.format = geodesic-patch/1` and a `geodesic_patch` JSON manifest.
- `patch_id` is SHA-256 of Python `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)` with `patch_id` removed. Go `encoding/json` escapes `&`, `<`, `>`; the decoder uses [`CanonicalPyJSON`](../../../x/weightedit/pyjson.go) instead.
- Checked-in fixture: [`x/weightedit/gpatch/testdata/`](../../../x/weightedit/gpatch/testdata/).

A new patch repo either emits this format or registers another `Codec`. There is no second first-party JSON patch format in v1.

## Identity claims

Bind verifies whatever the edit claims. Layout, config, and tokenizer claims always come from `edit.Source`, on both apply and restore (DAWE’s rule: sidecars are outside the weight fingerprint). Per-tensor hashes and the full fingerprint use the requested side. A DAWE edit’s full source fingerprint separates a fine-tune from its base when only embedding rows change. An edit that omits a claim does not invent one. There is no `fast` flag on the HTTP API.

After splicing, Apply rehashes the written directory when the edit includes a result (or source, on restore) fingerprint. `fingerprint_verified` is set only when that measurement matches.

## HTTP

| Method | Path | Behavior |
|--------|------|----------|
| POST | `/api/weights/edits/inspect` | Decode only |
| POST | `/api/weights/edits` | Decode, Bind(source), Apply, register destination |
| POST | `/api/weights/edits/restore` | Decode, Bind(result), Apply, register destination |

The server chooses the output directory (sibling of `source_dir`, name from `destination`). The request carries no filesystem path. Decoded patch cap **32 MiB** (JSON envelopes allow ~4/3 wire size for base64). Unknown format → **415**; invalid patch → **400**; fingerprint mismatch → **422**. Registration clones the source manifest and replaces `source_dir` on the config blob; Modelfile layers keep their digests.

## CLI

```bash
zerollama weights edit inspect PATCH
zerollama weights edit apply PATCH --model SRC --destination DST [--out DIR]
zerollama weights edit restore PATCH --model PATCHED --destination DST [--out DIR]
```

Offline; same library. `--out` overrides the sibling directory.

## Adding a codec

1. Implement `weightedit.Codec` (`Format`, `Detect`, `Decode`).
2. Call `weightedit.RegisterCodec` from `init`.
3. Decode must fill `Edit` with stored dtypes and old/new bytes (or leave resolution to Bind for a later delta/mask codec).
4. Add a fixture test that runs through Bind + Apply without knowing the decoder.

## Adding a store

Implement `weightedit.Store` for blob manifests or GGUF. Map physical layout to the logical shape on each `Change`. Codecs stay unchanged.

## Non-goals (v1)

- Searching for an edit (hidden-state export / ridge lab).
- Hot-swapping weights inside a loaded mlxrunner.
- Merging two edits of one base.
- LoRA adapters (existing adapter path).
