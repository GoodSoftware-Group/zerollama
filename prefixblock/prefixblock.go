// Package prefixblock implements L3 hash-chained prefix block ids
// (vLLM BlockPool-style content addressing). Must match Python
// runtime/kv/prefix_block_hash.py exactly — fleet LA13 / L3-R9 agents use
// these hashes in POST /api/fleet/assign prefix_block_hashes.
package prefixblock

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strings"
)

// RootHash is the 64-zero hex parent of block index 0.
const RootHash = "0000000000000000000000000000000000000000000000000000000000000000"

// ModelScope keys a prefix chain to one model layout (+ optional tenant salt).
func ModelScope(modelHash, cacheSalt string) string {
	modelHash = strings.TrimSpace(modelHash)
	salt := strings.TrimSpace(cacheSalt)
	if salt != "" {
		return modelHash + "\x00" + salt
	}
	return modelHash
}

// Hash returns SHA256(scope ‖ 0x00 ‖ parent ‖ index_be32 ‖ token_ids_be32signed) hex.
func Hash(scope, parentHash string, blockIndex int, tokens []int32) string {
	h := sha256.New()
	h.Write([]byte(scope))
	h.Write([]byte{0})
	h.Write([]byte(parentHash))
	var idx [4]byte
	binary.BigEndian.PutUint32(idx[:], uint32(blockIndex))
	h.Write(idx[:])
	var tok [4]byte
	for _, t := range tokens {
		binary.BigEndian.PutUint32(tok[:], uint32(t))
		h.Write(tok[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Block is one full prefix block in a chain.
type Block struct {
	Index      int
	Start      int
	End        int
	ParentHash string
	Hash       string
}

// Iter yields full blocks of blockSize tokens from tokens[0:].
func Iter(tokens []int32, blockSize int, scope string, maxTokens int) []Block {
	if blockSize <= 0 || len(tokens) == 0 {
		return nil
	}
	limit := len(tokens)
	if maxTokens > 0 && maxTokens < limit {
		limit = maxTokens
	}
	if limit <= 0 {
		return nil
	}
	parent := RootHash
	out := make([]Block, 0, limit/blockSize)
	idx := 0
	for start := 0; start+blockSize <= limit; start += blockSize {
		end := start + blockSize
		chunk := tokens[start:end]
		bh := Hash(scope, parent, idx, chunk)
		out = append(out, Block{
			Index:      idx,
			Start:      start,
			End:        end,
			ParentHash: parent,
			Hash:       bh,
		})
		parent = bh
		idx++
	}
	return out
}

// Hashes returns only the block hash strings from Iter (fleet assign helper).
func Hashes(tokens []int32, blockSize int, scope string, maxTokens int) []string {
	blocks := Iter(tokens, blockSize, scope, maxTokens)
	out := make([]string, len(blocks))
	for i, b := range blocks {
		out[i] = b.Hash
	}
	return out
}
