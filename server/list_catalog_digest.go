package server

import (
	"crypto/sha256"
	"encoding/hex"
)

// listCatalogDigest returns a stable 64-char hex digest for synthetic /api/tags rows
// (Eliza cloud stubs, LM Studio caches, etc.).
//
// Stock ollama clients (through at least 0.31.x) do digest[:12] unconditionally in
// `ollama ls` / `ollama ps` and panic on empty digests:
// https://github.com/ollama/ollama/issues/14250
func listCatalogDigest(seed string) string {
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:])
}
