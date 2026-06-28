package discover

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// ProbeSidecarArchitecture reads general.architecture from a draft sidecar GGUF.
func ProbeSidecarArchitecture(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("empty sidecar path")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return "", err
	}
	arch := strings.TrimSpace(meta.KV().Architecture())
	if arch == "" {
		return "", fmt.Errorf("sidecar missing architecture metadata")
	}
	return arch, nil
}

// DefaultProxyConvTensorForArch picks the lab conv proxy extract tensor for a sidecar arch.
func DefaultProxyConvTensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft":
		// blk.0.ffn_gate is square enough for top-left proxy; dflash_fc is fusion matmul not conv-shaped.
		return "blk.0.ffn_gate.weight"
	case "eagle3":
		return "blk.0.ffn_gate.weight"
	default:
		return DefaultProxyConvTensor()
	}
}

// DefaultProxyConvTensorForSidecar reads arch from path and returns the proxy tensor name.
func DefaultProxyConvTensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConvTensor(), "", err
	}
	return DefaultProxyConvTensorForArch(arch), arch, nil
}

// milSlotSpecForArch returns the tensor → MIL slot plan for a draft sidecar architecture.
func milSlotSpecForArch(arch string) []ANEDraftMILSlot {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft":
		return dflashMILSlotSpec
	default:
		return eagle3MILSlotSpec
	}
}

// milSlotSpecForDraft picks MIL slots from inventory spec_type and sidecar GGUF architecture.
func milSlotSpecForDraft(specType, sidecarArch string) []ANEDraftMILSlot {
	specType = strings.ToLower(strings.TrimSpace(specType))
	sidecarArch = strings.ToLower(strings.TrimSpace(sidecarArch))

	if sidecarArch == "dflash-draft" {
		return dflashMILSlotSpec
	}
	if specType == "dflash" && (sidecarArch == "qwen35" || sidecarArch == "qwen3") {
		return qwen35DrafterMILSlotSpec
	}
	return milSlotSpecForArch(sidecarArch)
}

func matchMILSlots(spec []ANEDraftMILSlot, tensors []*ggml.Tensor, proxyReady bool) (slots []ANEDraftMILSlot, matched, required int) {
	byName := make(map[string]*ggml.Tensor, len(tensors))
	for _, t := range tensors {
		if t != nil && t.Name != "" {
			byName[t.Name] = t
		}
	}

	out := make([]ANEDraftMILSlot, 0, len(spec))
	for _, slot := range spec {
		slot := slot
		if slot.MILPhase == "phase2_proxy" {
			slot.Ready = proxyReady
			if proxyReady {
				matched++
			}
			out = append(out, slot)
			continue
		}
		required++
		for _, pat := range slot.TensorPatterns {
			if t, ok := byName[pat]; ok {
				slot.Ready = true
				slot.MatchedTensor = t.Name
				slot.Shape = append([]uint64(nil), t.Shape...)
				matched++
				break
			}
		}
		out = append(out, slot)
	}
	return out, matched, required
}
