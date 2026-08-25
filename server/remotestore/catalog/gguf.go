// Package catalog indexes GGUF tensors into format-agnostic module roles.
//
// Why roles (not only raw names): future stream/runtime paging and MoE composition
// want layer.N.attn / layer.N.expert.K addressing that survives tokenizer/format
// quirks. v1 uses this for GET /v1/tensor; llama.cpp consumers are roadmap.
package catalog

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

// ModuleRole is a format-agnostic role tag for a weight tensor.
type ModuleRole string

const (
	RoleEmbed        ModuleRole = "embed"
	RoleNorm         ModuleRole = "norm"
	RoleLMHead       ModuleRole = "lm_head"
	RoleAttn         ModuleRole = "attn"          // layer.N.attn
	RoleFFN          ModuleRole = "ffn"           // layer.N.ffn
	RoleExpert       ModuleRole = "expert"        // layer.N.expert.K
	RoleSharedExpert ModuleRole = "shared_expert" // layer.N.shared_expert
	RoleOther        ModuleRole = "other"
)

// TensorEntry locates a tensor's bytes inside a content-addressed blob.
type TensorEntry struct {
	Name   string     `json:"name"`
	Role   ModuleRole `json:"role"`
	Digest string     `json:"digest"`
	Offset uint64     `json:"offset"` // absolute file offset
	Length int64      `json:"length"`
	Shape  []uint64   `json:"shape,omitempty"`
	Type   string     `json:"type,omitempty"`
}

var (
	reBlkAttn   = regexp.MustCompile(`^blk\.(\d+)\.attn`)
	reBlkFFNExp = regexp.MustCompile(`^blk\.(\d+)\.ffn_(?:gate|up|down)_exps?\.(\d+)`)
	reBlkFFN    = regexp.MustCompile(`^blk\.(\d+)\.ffn_`)
	reBlkShared = regexp.MustCompile(`^blk\.(\d+)\.ffn_.*shexp`)
	reBlkExpert = regexp.MustCompile(`^blk\.(\d+)\..*expert[s]?\.(\d+)`)
)

// RoleFromGGUFName maps a GGUF tensor name to a ModuleRole (+ optional layer/expert).
func RoleFromGGUFName(name string) (role ModuleRole, layer int, expert int) {
	layer, expert = -1, -1
	n := strings.ToLower(name)
	switch {
	case strings.HasPrefix(n, "token_embd") || strings.Contains(n, "embedding"):
		return RoleEmbed, -1, -1
	case strings.HasPrefix(n, "output_norm") || n == "output.weight" || strings.HasPrefix(n, "output."):
		if strings.Contains(n, "norm") {
			return RoleNorm, -1, -1
		}
		return RoleLMHead, -1, -1
	case strings.Contains(n, "norm"):
		if m := reBlkAttn.FindStringSubmatch(n); len(m) == 2 {
			layer, _ = strconv.Atoi(m[1])
			return RoleNorm, layer, -1
		}
		return RoleNorm, -1, -1
	}

	if m := reBlkFFNExp.FindStringSubmatch(n); len(m) == 3 {
		layer, _ = strconv.Atoi(m[1])
		expert, _ = strconv.Atoi(m[2])
		return RoleExpert, layer, expert
	}
	if m := reBlkExpert.FindStringSubmatch(n); len(m) == 3 {
		layer, _ = strconv.Atoi(m[1])
		expert, _ = strconv.Atoi(m[2])
		return RoleExpert, layer, expert
	}
	if reBlkShared.MatchString(n) {
		if m := reBlkFFN.FindStringSubmatch(n); len(m) == 2 {
			layer, _ = strconv.Atoi(m[1])
		}
		return RoleSharedExpert, layer, -1
	}
	if m := reBlkAttn.FindStringSubmatch(n); len(m) == 2 {
		layer, _ = strconv.Atoi(m[1])
		return RoleAttn, layer, -1
	}
	if m := reBlkFFN.FindStringSubmatch(n); len(m) == 2 {
		layer, _ = strconv.Atoi(m[1])
		return RoleFFN, layer, -1
	}
	return RoleOther, -1, -1
}

// RoleString formats role with layer/expert indices for addressing.
func RoleString(role ModuleRole, layer, expert int) string {
	switch role {
	case RoleAttn:
		if layer >= 0 {
			return fmt.Sprintf("layer.%d.attn", layer)
		}
	case RoleFFN:
		if layer >= 0 {
			return fmt.Sprintf("layer.%d.ffn", layer)
		}
	case RoleExpert:
		if layer >= 0 && expert >= 0 {
			return fmt.Sprintf("layer.%d.expert.%d", layer, expert)
		}
	case RoleSharedExpert:
		if layer >= 0 {
			return fmt.Sprintf("layer.%d.shared_expert", layer)
		}
	case RoleEmbed, RoleNorm, RoleLMHead:
		return string(role)
	}
	return string(role)
}

// CatalogGGUF opens a GGUF file and returns tensor entries keyed by name and role string.
func CatalogGGUF(path, digest string) ([]TensorEntry, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dataBase := uint64(f.TensorDataOffset())
	var out []TensorEntry
	for _, ti := range f.TensorInfos() {
		role, layer, expert := RoleFromGGUFName(ti.Name)
		out = append(out, TensorEntry{
			Name:   ti.Name,
			Role:   ModuleRole(RoleString(role, layer, expert)),
			Digest: digest,
			Offset: dataBase + ti.Offset,
			Length: ti.NumBytes(),
			Shape:  append([]uint64(nil), ti.Shape...),
			Type:   fmt.Sprintf("%v", ti.Type),
		})
	}
	return out, nil
}

// LookupGGUF finds a tensor by raw name or module-role string.
func LookupGGUF(path, digest, ref string) (TensorEntry, error) {
	entries, err := CatalogGGUF(path, digest)
	if err != nil {
		return TensorEntry{}, err
	}
	ref = strings.TrimSpace(ref)
	for _, e := range entries {
		if e.Name == ref || string(e.Role) == ref {
			return e, nil
		}
	}
	// prefix match on role for multi-tensor roles (e.g. layer.0.attn matches all attn tensors)
	var matches []TensorEntry
	for _, e := range entries {
		if strings.HasPrefix(string(e.Role), ref) || strings.HasPrefix(e.Name, ref) {
			matches = append(matches, e)
		}
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) > 1 {
		return TensorEntry{}, fmt.Errorf("ambiguous tensor ref %q (%d matches); use exact GGUF name", ref, len(matches))
	}
	return TensorEntry{}, fmt.Errorf("tensor %q not found", ref)
}
