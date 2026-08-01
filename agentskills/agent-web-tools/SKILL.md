---
name: agent-web-tools
description: "Use zerollama's experimental server-side web search and web fetch endpoints as agent tools."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, web-search, web-fetch, tools, experimental, cloud-proxy]
    category: mlops
    related_skills: [zerollama-integration]
---

# Agent Web Tools Skill

Give a local agent web search and page-fetch capability through a
[zerollama](https://github.com/GoodSoftware-Group/zerollama) server's experimental
proxy endpoints, without wiring a separate search API key into the agent
itself. Both endpoints proxy to a cloud backend (Eliza Cloud) — they are
**not** local web crawling.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/experimental/web_search   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/web_search   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/experimental/web_fetch   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/web_fetch   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- An agent loop needs a "search the web" or "fetch this URL" tool and you
  want to route it through zerollama's proxy instead of a bespoke provider
  integration
- Debugging a `web_search`/`web_fetch` tool call failing with a
  cloud-unavailable error

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- The underlying cloud proxy must be reachable/configured server-side —
  these are **experimental** endpoints and return an explicit
  cloud-unavailable error if the proxy isn't configured; there's no local
  fallback

## API Contract

| Endpoint | Method | Notes |
|---|---|---|
| `/api/experimental/web_search` | `POST` | Proxies to cloud `/api/web_search`; body/response shape mirrors the upstream provider |
| `/api/experimental/web_fetch` | `POST` | Proxies to cloud `/api/web_fetch`; fetches and returns page content for a given URL |

Both require a non-empty JSON body — an empty body returns `400 missing
request body`.

## How to Run

```bash
# Web search
curl -s -X POST http://localhost:11434/api/experimental/web_search \
  -H 'Content-Type: application/json' \
  -d '{"query": "zerollama vram broker design"}'

# Web fetch (retrieve a specific page)
curl -s -X POST http://localhost:11434/api/experimental/web_fetch \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/some-article"}'
```

## Pitfalls

- **`experimental` in the path means the contract can change** — don't
  build a hard dependency on exact response field names without checking
  the current server version.
- **Errors mean cloud proxy unavailability, not a bad query** — a failure
  here (`cloudErrWebSearchUnavailable` / `cloudErrWebFetchUnavailable`)
  usually means the cloud proxy isn't reachable/configured on this
  zerollama instance, not that your search terms or URL were malformed.
- **Not a local crawler** — these calls leave the box; don't use them for
  air-gapped/offline agent setups. There is no local-only search backend.
- **Empty body is a hard 400** — always send a JSON body even for a
  "default" search; there's no bare-query-string form.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
