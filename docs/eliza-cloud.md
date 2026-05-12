# Eliza Cloud (Zerollama)

Remote inference uses [Eliza Cloud](https://www.elizacloud.ai) by default. This is **not** the legacy ollama.com cloud API.

**Why a separate default upstream:** Zerollama targets agents and integrations that expect **OpenAI- and Anthropic-compatible** HTTP APIs. Eliza exposes those under `/api/v1/...` with API-key auth. The historical Ollama cloud stack (Ed25519-signed requests to `ollama.com`) remains available as an **opt-in** base URL for users who still depend on it.

## Configuration

| Variable | Purpose |
|----------|---------|
| `ELIZACLOUD_API_KEY` | Organization API key; sent as `X-API-Key` on **all** outbound proxied requests to non-`ollama.com` hosts (inference, catalog, experimental routes). If unset, the server logs **once** per process that requests may return 401 until the key is configured. |
| `OLLAMA_CLOUD_BASE_URL` | Optional override of the cloud base URL (default `https://www.elizacloud.ai:443`). Set `https://ollama.com:443` only if you rely on the legacy Ollama cloud **signing** flow. |
| `OLLAMA_NO_CLOUD` | Disables remote cloud features entirely (no catalog merge, no proxy). |

**Why `X-API-Key`:** Eliza’s documented auth model is API-key based. Stripping conflicting client `Authorization` / wallet headers on outbound Eliza calls avoids accidentally forwarding browser or tool credentials that Eliza would reject or misinterpret.

**Why Ed25519 only for `ollama.com`:** Request signing is part of the **legacy** Ollama cloud contract. Eliza does not use that challenge/signature scheme; attaching signatures to Eliza would be meaningless and could break requests.

## Model names

Use Eliza catalog model ids with the Zerollama **`:cloud`** suffix, for example `openai/gpt-4o-mini:cloud`. The suffix marks the model as **remote** in the local registry; it is stripped before the upstream `model` field is sent to Eliza.

**Why a suffix instead of a separate namespace only:** Existing UIs and CLIs already reason about “model names.” The `:cloud` source tag keeps one mental model: same string shape as local tags, with an explicit routing bit the server can parse without a second registry.

## Path mapping

Client-facing routes stay familiar (`/v1/chat/completions`, etc.). For Eliza, the proxy **rewrites** paths (only the real OpenAI v1 prefix: `/v1` or `/v1/...`, not arbitrary paths starting with `/v1`):

- `/v1` or `/v1/*` → `/api/v1` or `/api/v1/*`
- `/api/embed` and `/api/embeddings` → `/api/v1/embeddings`

**Why rewrite instead of exposing `/api/v1` only:** Tools and libraries already target `/v1/...` and `/api/embed*`. Rewriting at the proxy preserves compatibility while matching Eliza’s actual route tree.

## Catalog merge

`GET /api/tags` (and related listing) **merges** local models with models returned from `GET {base}/api/v1/models` when cloud is enabled.

**Why merge:** Users should see **one** list: local GGUFs and remote Eliza ids, distinguished by metadata (`Details.Family: cloud`, `RemoteModel`, etc.).

**Why singleflight + TTL cache:** Concurrent requests should not stampede the upstream catalog. **`singleflight`** deduplicates in-flight fetches. **`Cache-Control`** (`s-maxage` / `max-age`) sets refresh interval when present, clamped to a sane range; otherwise a default TTL applies so we do not hammer Eliza on every UI poll.

**Why dedupe by lowercased `Model`:** The same logical id can appear as a manually added `:cloud` entry and again from the remote list; keeping one row avoids duplicate picks in the UI.

## APIs to use for cloud models

- Prefer **OpenAI-compatible** `POST /v1/chat/completions`, `POST /v1/embeddings`, and **Anthropic-compatible** `POST /v1/messages` with a `:cloud` model.
- Native `POST /api/chat` and `POST /api/generate` are **not** used for Eliza cloud models; the server returns an error that points you at the v1 routes.

**Why:** Eliza speaks OpenAI/Anthropic shapes. Bridging everything through the legacy Ollama chat/generate pipeline would duplicate work and blur error semantics.

## Responses that are “raw Eliza” JSON

Some endpoints forward **upstream JSON as-is** (for example Eliza-backed `GET /v1/models/:model` and certain show paths). They are **not** re-shaped into Ollama’s `api.ShowResponse` or OpenAI’s `ToModel` types.

**Why:** Metadata schemas differ between hosts. Reshaping without a guaranteed stable mapping would silently drop fields or lie about capabilities. Raw passthrough keeps the client aligned with what Eliza actually returns until a deliberate mapping exists.

## Account endpoints

`POST /api/me` and `POST /api/signout` target **ollama.com** only when `OLLAMA_CLOUD_BASE_URL` points at `ollama.com`. For Eliza, `/api/me` returns a small JSON stub describing Eliza Cloud usage so the UI does not assume an Ollama.com session.

**Why:** Those routes implement the **hosted Ollama** account contract. Eliza accounts are a different system; pretending they are the same would confuse auth and logout behavior.

## See also

- [CHANGELOG.md](../CHANGELOG.md) — release notes for cloud-related changes.
- [Roadmap — remote cloud](ROADMAP.md#zerollama-remote-cloud-eliza) — planned follow-ups and non-goals.
