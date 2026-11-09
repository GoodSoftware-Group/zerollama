# Invariants

These are the contracts the scheduler, storage, and the Go–runtime wire rely on. The check is a named function. Call sites use that function. A later change that re-derives the condition inline is how the rule disappears.

The functions and their tests are the source of truth:

| Rule | Function | Test |
| --- | --- | --- |
| Unset `OLLAMA_MAX_LOADED_MODELS` is 3 models per GPU, applied before eviction | `server.MaxRunnersForGPUs` | `TestInvariantMaxRunnersUnsetIsThreePerGPU` |
| `requireFull` fails when the VRAM estimate exceeds free memory | `llm.OffloadFits` | `TestInvariantOffloadMustCoverEstimate` |
| Unified memory caps the 262144 default context at 32768 | `server.DefaultNumCtx` | `TestInvariantIntegratedMemoryCapsDefaultContext` |
| Sidecar `/health` 200 counts only when the body has `server_revision` | `server.IsZerollamaRuntimeHealth` | `TestInvariantRuntimeHealthRequiresOurBody` |
| Gallery manifest, blob, and transfer requests send `ollama/<OllamaCompat>` | `server.GalleryUserAgent` | `TestInvariantGalleryUserAgentIsOllamaCompat` |
| Split GGUF symlinks stay one file per shard name | `ggufSplitBasesCollapse` | `TestInvariantSplitNamesMustBeUnique` |
| Reclaim deletes a local blob only when remote size equals local size | `remotestore.RemoteBlobMatchesLocal` | `TestInvariantRemoteMatchIsSizeNotStatus` |
| A zero-byte digest path is not a cache hit | `remotestore.LocalBlobComplete` | `TestInvariantZeroByteBlobIsNotComplete` |
| GC skips a blob younger than an hour | `remotestore.BlobTooYoungToGC` | `TestInvariantYoungBlobSurvivesGC` |
| Ephemeral delete waits until the pin count is zero | `remotestore.EphemeralMayDelete` | `TestInvariantSharedPinBlocksEphemeralDelete` |
| `storage pull` succeeds only after a `success` event | `remotestore.PullStreamComplete` | `TestInvariantPullNeedsSuccessEvent` |
| `POST /v1/pull` writes the tree this process serves | `remotestore.PullWritesServedTree` | `TestInvariantPullWritesTheServedTree` |
| Client blob paths are `sha256-` plus 64 lowercase hex | `safeBlobDigest` | `TestInvariantClientDigestIsHexOnly` |
| Omitted `max_tokens` is unlimited, and admission stores 0 extra tokens | `completion_cap`, `admit_token_budget` | `test_omitted_cap_is_unlimited` |
| `think: false` stays on the runtime | `think_forces_legacy` | `test_think_false_stays_on_runtime` |
| Unknown loaded context reloads | `loaded_ctx_must_grow` | `test_unknown_loaded_context_must_reload` |
| `tool_choice` filters the tool list | `apply_tool_choice` | `test_tool_choice_filters` |
| Omitted `keep_alive` refreshes the lease to the server default | `server.KeepAliveDuration` | `TestInvariantOmittedKeepAliveIsTheDefault` |
| An in-flight load is joined only when context, shift, adapters, and runner options match, and the load keeps the process until it finishes | `server.LoadingCannotJoin`, `Scheduler.deferToInFlightLoad` | `TestInvariantInFlightContextMustCoverTheRequest`, `TestInFlightLoadIsNotExpired` |
| A ggml load releases the runtime even when the weights fit | `server.KeepRuntimeResident` | `TestInvariantGGMLLoadReleasesTheRuntime` |
| Metal, CPU, and integrated memory do not poll for VRAM recovery | `server.VRAMRecoveryNeeded` | `TestInvariantVRAMRecoverySkipsUnifiedMemory` |
| Eviction takes the shorter keep-alive, then the least recently used | `evictBefore` | `TestInvariantShorterLeaseEvictsFirst` |
| A negative `keep_alive` stays the maximum duration | `server.KeepAliveDuration` | `TestInvariantNegativeKeepAliveStaysForever` |
| A gallery 200 is the blob URL, and a redirect uses Location | `server.DirectBlobURL` | `TestInvariantGallery200IsTheBlobURL` |
| A gallery 200 is stored only when the body is the whole object. A later part rejects a 200 of the whole file | `server.InlineBlobComplete`, `server.RangeBodyFitsPart` | `TestInvariantShortGalleryBodyIsNotTheBlob` |
| A digest file matches the manifest size and the sha256 of its bytes | `remotestore.BlobContentTrusted` | `TestInvariantSameSizeCorruptBlobIsNotTrusted` |
| Cache and models blob directories match by path or symlink | `remotestore.SameBlobDir` | `TestInvariantBlobDirSymlinkIsTheModelsDir` |
| A parsed local tag is replaced when storage has different manifest bytes. A failed check expires sooner than a success | `remotestore.ManifestCheckFresh` | `TestInvariantFailedManifestCheckExpiresSooner` |
| `GET /slots` length is the fitted `-np` | `llamaCompatSlots` | `TestLlamaCompatSlotsUsesFittedParallel` |
| A hot model may skip older work at most twice | `PopPreferringKeys` | `TestPendingQueueHotPreferIsBounded` |
| Cache eviction on the models tree skips digests a manifest names | `remotestore.ManifestReferencedDigests` | `TestInvariantManifestDigestsAreNotEvictableNames` |
| A runtime stream commits HTTP 200 only after the sidecar accepts | `forwardRuntimeNDJSON` | `TestRuntimeGenerateStreamKeepsSidecarStatus` |
| Runtime listen and llama-server `port+1` stay off 11434, 8081, and 18090 | `runtime_port_conflict` | `test_runtime_ports_stay_off_reserved_listeners` |

`options.gguf` on a proxied request is the manifest blob for that tag, or it is absent. `TestRuntimeProxyOptionsDropsClientGguf` locks that.

A paused load, a timed-out eviction, and a FIFO yield requeue or rearm instead of starting the next load. Those paths call `requeueAfter` and the delayed notify in `pendingPopNext`.

## Accepted on purpose

These are the rules the code implements. They are not fail-closed checks.

- A model is a tag path. There is no `@digest` pin, so the next pull of that tag replaces the weights.
- Lookup folds case. The directory keeps the spelling of the first write.
- A chat template may snap to the nearest builtin within edit distance 100.
- A name ending in `-cloud` is the Eliza stub.
- A blob PUT signature covers method, path, and time. The server hashes the body. Multi-gigabyte uploads are not HMAC'd twice.
- Storage manifest responses are not signed, and mDNS TXT is not a peer identity.
- A passing digest hash is remembered for that size and mtime. A later load with the same stamp does not read the file again.
- A successful storage check stands for 5 seconds. A failed check stands for 2 seconds, and the local tag is still served.
