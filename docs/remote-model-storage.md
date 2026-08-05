# Remote model storage

Centralize GGUF (and other Ollama-shaped) model blobs on one or more storage servers; inference nodes fetch on miss into a local cache.

**Why this exists:** Operators often accumulate hundreds of models (hundreds of GB) on the inference box. Disk fills up long before VRAM/CPU do. Replicating every blob to every node wastes space; a single canonical tree on a bigger disk, with on-demand pull, matches how models are already content-addressed (`sha256-…` digests under `$OLLAMA_MODELS/blobs`).

**Why not NFS/S3 alone:** POSIX mounts and object stores are fine for cold archives, but they do not speak Ollama manifests, HMAC between zerollama peers, InfiniBand preference, or the tensor-addressed language we need for later streaming. A small daemon next to the models tree keeps the wire close to how zerollama already names blobs.

Full operator guide: this file. Package: `server/remotestore/`. CLI: `zerollama storage serve|push`.

---

## Problem → design

| Constraint | Design choice | Why |
|------------|---------------|-----|
| Disk full on inference hosts | Central `storage serve` + fetch-on-miss in `GetModel` | Move canonical tree once; nodes keep only what they use |
| Blobs already SHA-256 addressed | Content-addressed `/v1/blob/{digest}` | Integrity and dedup come free; shared layers upload once |
| Trusted LAN / IB fabric, not public internet | HMAC-SHA256 shared secret, no TLS by default | Cheap auth on LAN; TLS is an ops add-on when the path leaves the fabric |
| IB often faster than TCP when available | Prefer RDMA capability, fall back to TCP Range-GET | Use the fabric you paid for without requiring verbs on every build |
| Must not kill loaded runners under LRU | Refcounted `Pin`/`Unpin` from the scheduler | Evicting an mmap’d blob → SIGBUS / open failures |
| Shared layers across models | Pin **refcount**, not a boolean set | Unloading model A must not unpin a digest still held by model B |
| `--reclaim` after migration | Delete only after **all** referencing manifests pushed | Mid-walk delete of shared blobs leaves remote manifests without layers |
| Corrupt transfer during download | Hash while writing `.partial`, rename only on match | A crash between rename and verify must not leave a “good-looking” bad file |
| Concurrent `GetModel` for same digest | `singleflight` per digest | Two writers truncating the same `.partial` corrupt the cache |
| Future stream / MoE / KV reuse | Catalog roles + `tensorproto` + payload-agnostic auth/transport | Spec the language now; do not ship llama.cpp patches in v1 |

---

## Quick start

### Storage server

```bash
export ZEROLLAMA_STORAGE_SECRET='long-random-shared-secret'
export OLLAMA_MODELS=/data/models   # canonical tree
./zerollama storage serve --listen 0.0.0.0:18090
```

Lab default listen is `:18090`. **Why not `:11434` / `:8081`:** those are production inference / runtime ports on operator machines; agents and docs must never compete with them.

### Migrate blobs from an inference host

```bash
export ZEROLLAMA_STORAGE_SECRET='long-random-shared-secret'
./zerollama storage push http://storage-host:18090
# after you verify remote HEAD/GET looks right:
./zerollama storage push http://storage-host:18090 --reclaim
```

**Why two-phase reclaim:** push walks many manifests; one blob can appear in dozens. Deleting after the first successful upload breaks later manifests that still need the local file to upload (or to skip HEAD). Reclaim runs only when every local reference has a successful remote push.

### Inference node

```bash
export ZEROLLAMA_STORAGE_SERVERS=http://storage-host:18090
export ZEROLLAMA_STORAGE_SECRET='long-random-shared-secret'
export ZEROLLAMA_REMOTE_CACHE_MAX_BYTES=$((200*1024*1024*1024))  # optional LRU cap
# optional: ZEROLLAMA_REMOTE_CACHE_MODE=ephemeral
./zerollama serve   # ordinary serve; GetModel fetches missing blobs transparently
```

**Why integrate into `GetModel`:** every load path already goes through manifests + layer digests. Transparent miss-fetch means `zerollama run` / `/api/chat` keep working without a separate “sync models” step.

---

## Security posture (trusted LAN)

v1 uses **HMAC-SHA256** request signing:

```http
Authorization: HMAC-SHA256 <unix_ts>.<hex_hmac>
```

MAC input: `"<ts>\n<METHOD>\n<path>\n<body_sha256_hex>"`. Replay window ≈ 5 minutes.

| Choice | Why |
|--------|-----|
| Shared secret, not mTLS in v1 | Fastest path for a closed fabric; secret via env or `_FILE` |
| No TLS by default | LAN/IB already isolated; TLS when crossing untrusted nets (reverse proxy or `http.Server` certs) |
| Large blob PUT signs **empty** body | Streaming multi-GB uploads without buffering the body for HMAC; integrity is the digest in the URL + server hash-while-write |
| Digests must be `sha256-<64 hex>` | Prevents authenticated path traversal (`sha256-../../../etc/passwd`) |

HMAC stops unauthenticated reads/writes and casual replay outside the window. It does **not** encrypt weights in transit.

---

## Transports

| Priority | Transport | Status | Why this order |
|----------|-----------|--------|----------------|
| 1 | RDMA READ (InfiniBand/RoCE verbs) | **First-class** when both binaries are built with `-tags rdma` and an HCA is active. Control: HMAC `POST /v1/rdma/session` + `POST /v1/rdma/mr`. Data: one-sided `IBV_WR_RDMA_READ`. | Lowest latency / highest BW on IB fabrics |
| 2 | TCP/HTTP Range-GET | Always available | Universal fallback; Ethernet and IPoIB |

Control plane (capability, manifests, auth, RDMA session/MR lease) is always TCP/HTTP. **Why:** QP setup and MR leases are small JSON; only bulk bytes belong on verbs.

**Build:** `CGO_ENABLED=1 go build -tags rdma -o zerollama .` (needs `libibverbs-dev`). Capability includes `"verbs": true` only when storaged opened an HCA. Clients require `verbs:true` before selecting RDMA — otherwise TCP (no fake `via=rdma`).

**Server MR note:** mlx4 has no ODP, so file-mmap `ibv_reg_mr` fails (EFAULT). Storaged uses a **bounce buffer** (ReadAt into a recycled C-heap MR). The wire path is still RDMA READ.

**Throughput expectations (mlx4 QDR 40 Gb/s):** the bounce + per-window pin dominates, not the IB link. Pipeline outstanding READs (`max_rd_atomic` up to 16, negotiated in the session) and pin-prefetch help, but first-fetch rates are typically hundreds of MiB/s unless the peer is ODP-capable (mlx5) or the blob is already resident in a long-lived MR. Redeploy **both** client and `storage serve` with the same `-tags rdma` binary so responder depth matches.

Roadmap transports: raw Ethernet L2 (`AF_PACKET`), UDP/ARQ; ODP/file-backed MRs.

---

## Cache modes

| Mode | Env | Behavior | Why |
|------|-----|----------|-----|
| `persist` (default) | `ZEROLLAMA_REMOTE_CACHE_MODE=persist` | Blobs under `ZEROLLAMA_REMOTE_CACHE_DIR` or `$OLLAMA_MODELS/blobs`; LRU when over `ZEROLLAMA_REMOTE_CACHE_MAX_BYTES` | Warm restarts without re-fetch; cap protects the boot disk |
| `ephemeral` | `=ephemeral` | Scratch dir; deleted when the runner unloads | Zero persistent footprint for one-shot loads |

**Why pin from the scheduler:** LRU must not delete digests still mmap’d or open by a loaded runner. Pins are taken at runner register and released on unload (including orphan/replace paths).

**Why manifests always under `$OLLAMA_MODELS`:** `ParseNamedManifest` reads that tree. Blob cache dir can differ; if manifests followed the cache dir, fetch would “succeed” and `GetModel` would still fail to parse.

---

## API surface (`zerollama storage serve`)

| Endpoint | Why |
|----------|-----|
| `GET /v1/capability` | Negotiate RDMA vs TCP (`verbs:true` when QP path is live) |
| `POST /v1/rdma/session` | Exchange RC QP endpoints (HMAC JSON) |
| `POST\|DELETE /v1/rdma/mr` | Lease / release a blob-range MR for RDMA READ |
| `HEAD\|GET /v1/blob/{sha256-…}` | Content-addressed bulk; Range-GET fallback |
| `PUT /v1/blob/{sha256-…}` | Migration / replication; stream + hash |
| `GET\|PUT /v1/manifest/{host}/{ns}/{model}/{tag}` | Same layout as local manifests |
| `GET /v1/tensor/{host}/{ns}/{model}/{tag}/{tensor_ref}` | Tensor-addressed convenience over byte ranges |

All require HMAC.

**Why two layers (bytes + tensors):** runners and migration need digests and ranges. Future stream/runtime paging wants names/roles (`layer.3.attn`). Server-side catalog resolve keeps clients from re-implementing GGUF offset math.

---

## Tensor / catalog language

`server/remotestore/catalog` maps GGUF tensor names → module roles (`embed`, `layer.N.attn`, `layer.N.ffn`, `layer.N.expert.K`, …).

`server/remotestore/tensorproto` is the **spec** for per-tensor fetch (request/response/error codes). **Why ship the spec without llama.cpp patches:** so stream / runtime-cache work later speaks one language; v1 stays shippable without C++ risk.

---

## Env reference

| Variable | Role | Why |
|----------|------|-----|
| `ZEROLLAMA_STORAGE_SERVERS` | Comma-separated base URLs | Multi-server fallback |
| `ZEROLLAMA_STORAGE_SECRET` / `_FILE` | HMAC secret | Shared with `storage serve` |
| `ZEROLLAMA_STORAGE_LISTEN` | Serve bind (default `0.0.0.0:18090`) | Lab port, not inference |
| `ZEROLLAMA_REMOTE_CACHE_MODE` | `persist` \| `ephemeral` | Footprint policy |
| `ZEROLLAMA_REMOTE_CACHE_DIR` | Blob cache root (default `$OLLAMA_MODELS`) | Optional separate SSD |
| `ZEROLLAMA_REMOTE_CACHE_MAX_BYTES` | LRU cap (`0` = unlimited) | Protect boot disk |
| `ZEROLLAMA_REMOTE_SCRATCH_DIR` | Ephemeral scratch | Override temp |

---

## Cluster fabric note

Auth + `BulkTransport` are **payload-agnostic**. **Why:** the next consumers (remote KV spillover, compute buffer move) should reuse HMAC and transport preference instead of inventing a second wire. A package rename to a generic “fabric” waits until a second consumer ships.

---

## Roadmap (explicitly deferred)

- ODP / file-backed MRs (skip bounce copy on capable NICs)
- Raw L2 / UDP transports
- llama.cpp `stream` (`llama_model_init_from_user`) and runtime tensor cache
- Cross-model MoE composition tooling
- Remote KV-cache spillover
- Multi-node compute sharing (tensor/pipeline parallel)

---

## Build notes

```bash
# TCP-only (default) — why: most hosts have no libibverbs
go build -o zerollama .

# With RDMA preference (needs libibverbs-dev / librdmacm-dev)
CGO_ENABLED=1 go build -tags rdma -o zerollama .
```

---

## Code map

| Path | Role |
|------|------|
| `server/remotestore/` | Client resolver, auth, transports |
| `server/remotestore/storaged/` | HTTP server for `storage serve` |
| `server/remotestore/catalog/` | GGUF → module-role index |
| `server/remotestore/tensorproto/` | Spec-only tensor fetch types |
| `cmd/storage.go` | `zerollama storage serve\|push` |
| `server/images.go` | `ensureBlob` / `GetModel` miss-fetch |
| `server/sched.go` | Pin on load, `ReleaseModelBlobs` on unload |
