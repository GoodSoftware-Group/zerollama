---
name: account-auth
description: "Check zerollama/Eliza Cloud account identity and manage local signing keys via /api/me, /api/signout, and /api/user/keys."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, auth, eliza-cloud, whoami, signout, keys]
    category: mlops
    related_skills: [zerollama-integration, download-model, cloud-model-routing]
---

# Account / Auth Skill

Check which cloud account (if any) a local [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server is signed in as, and manage its local signing key. This only matters
when the server proxies some traffic to Eliza Cloud or `ollama.com` (push,
cloud model fallback); pure-local inference doesn't require any of this.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/me   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/signout   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/user/keys   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/user/keys/:encodedKey   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Confirming which account is authenticated before a `push`/cloud-model
  call
- Signing out / rotating the local key on this host
- Debugging push/pull failures against a private registry that need auth

## API Contract

| Endpoint | Method | Notes |
|---|---|---|
| `/api/me` | `POST` | Whoami. Behavior depends on the configured cloud host: against `ollama.com` it returns the authenticated account; against any other Eliza Cloud host it returns proxy info (`cloud_host`, docs pointer) instead of a real identity |
| `/api/signout` | `POST` | Signs out of `ollama.com` using the local key; against a non-`ollama.com` Eliza Cloud host it's a no-op (returns "sessions are managed in the Eliza Cloud dashboard") |
| `/api/user/keys/:encodedKey` | `DELETE` | Alias for sign-out scoped to a specific base64url-encoded public key |

## How to Run

```bash
# Who am I signed in as?
curl -s -X POST http://localhost:11434/api/me

# Sign out (clears the local key's association with ollama.com)
curl -s -X POST http://localhost:11434/api/signout
```

## Pitfalls

- **`/api/me` doesn't mean "is inference configured"** — it's purely cloud
  account identity; local GGUF chat/embedding/image inference works with no
  account at all. Don't gate local-only agent setups on this returning a
  real identity.
- **Behavior forks on the configured cloud host** — if
  `ELIZACLOUD_API_KEY`/cloud proxy points somewhere other than
  `ollama.com`, `/api/me` and `/api/signout` return generic
  proxy/no-op responses instead of real account state. Check
  `cloud_host` in the response before assuming sign-out failed.
- **No local session to "log back into"** — the local key is a keypair on
  disk (`auth.GetPublicKey()`), not a session token; "signing out" affects
  the *remote* association, not local key material.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `download-model` — `push`/private-registry pulls that this identity gates
- `cloud-model-routing` — actually calling remote Eliza Cloud models once authenticated
