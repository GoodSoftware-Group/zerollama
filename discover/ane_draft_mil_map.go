package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// ANEDraftMILSlot maps an Eagle3 GGUF tensor role to a future ANE MIL weight slot.
type ANEDraftMILSlot struct {
	Slot           string   `json:"slot"`
	Role           string   `json:"role"`
	TensorPatterns []string `json:"tensor_patterns"`
	MILPhase       string   `json:"mil_phase"`
	Ready          bool     `json:"ready"`
	MatchedTensor  string   `json:"matched_tensor,omitempty"`
	Shape          []uint64 `json:"shape,omitempty"`
	Note           string   `json:"note,omitempty"`
}

// ANEDraftMILMapResult is the Eagle3 tensor → MIL compile plan.
type ANEDraftMILMapResult struct {
	OK                  bool               `json:"ok"`
	Mode                string             `json:"mode"`
	Tag                 string             `json:"tag,omitempty"`
	SpecType            string             `json:"spec_type,omitempty"`
	DraftSidecarPresent bool               `json:"draft_sidecar_present"`
	DraftGGUF           string             `json:"draft_gguf,omitempty"`
	DraftSearchPaths    []string           `json:"draft_search_paths,omitempty"`
	ProxyChannels       int                `json:"proxy_channels"`
	ProxySpatial        int                `json:"proxy_spatial"`
	WeightBlobBytes     int                `json:"weight_blob_bytes"`
	Slots               []ANEDraftMILSlot  `json:"slots"`
	MatchedCount        int                `json:"matched_count"`
	RequiredCount       int                `json:"required_count"`
	Blockers            []string           `json:"blockers,omitempty"`
	NextStep            string             `json:"next_step,omitempty"`
	Note                string             `json:"note,omitempty"`
}

// eagle3MILSlotSpec is the static mapping table from llama.cpp eagle3 arch.
var eagle3MILSlotSpec = []ANEDraftMILSlot{
	{
		Slot:           "encoder_fc",
		Role:           "fuse target layer hiddens → draft hidden (llama_encode)",
		TensorPatterns: []string{"fc.weight"},
		MILPhase:       "phase3_subgraph",
		Note:           "matmul [3*n_embd_tgt × n_embd_dec]; not the lab conv proxy",
	},
	{
		Slot:           "decoder_attn_q",
		Role:           "draft decoder attention Q projection",
		TensorPatterns: []string{"blk.0.attn_q.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "decoder_ffn_gate",
		Role:           "draft decoder FFN gate",
		TensorPatterns: []string{"blk.0.ffn_gate.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "proxy_conv_w0",
		Role:           "lab ANE conv proxy (latency stand-in for full draft step)",
		TensorPatterns: []string{},
		MILPhase:       "phase2_proxy",
		Note:           "synthetic fp16 or sidecar extract via ane-draft-mil-extract → --weight-file",
	},
}

// ANEDraftSidecarCandidates returns known drafter GGUF search paths for a tag.
func ANEDraftSidecarCandidates(shortName string) []string {
	shortName = strings.TrimSpace(shortName)
	if shortName == "" {
		return nil
	}
	home := os.Getenv("HOME")
	if home == "" {
		home = "/"
	}
	lower := strings.ToLower(shortName)
	out := []string{
		filepath.Join(home, ".ollama", "models", "draft", shortName),
		filepath.Join(home, "Models", "draft", shortName),
	}
	switch {
	case strings.Contains(lower, "2b") && strings.Contains(lower, "dflash"):
		out = append(out, filepath.Join(home, ".cache", "zerollama", "eliza-1", "bundles", "2b", "dflash", "drafter-2b.gguf"))
	case strings.Contains(lower, "27b") && strings.Contains(lower, "dflash"):
		out = append(out, filepath.Join(home, ".cache", "zerollama", "eliza-1", "bundles", "27b-256k", "dflash", "drafter-27b-256k.gguf"))
	}
	return out
}

func resolveDraftGGUFPath(entry ANEDraftEntry) (path string, present bool) {
	if p := strings.TrimSpace(entry.DraftGGUF); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, true
		}
	}
	short := strings.SplitN(entry.Tag, ":", 2)[0]
	if p := FindANEDraftSidecarPath(short); p != "" {
		return p, true
	}
	candidates := ANEDraftSidecarCandidates(short)
	if len(candidates) > 0 {
		return candidates[0], false
	}
	return "", false
}

func draftMILWeightBlobBytes(channels int) int {
	if channels <= 0 {
		channels = 64
	}
	// Matches tools/ane-draft/draft_bench.m buildDraftWeightBlob layout.
	return 64 + 64 + channels*channels*2
}

func matchEagle3MILSlots(tensors []*ggml.Tensor, proxyReady bool) (slots []ANEDraftMILSlot, matched, required int) {
	byName := make(map[string]*ggml.Tensor, len(tensors))
	for _, t := range tensors {
		if t != nil && t.Name != "" {
			byName[t.Name] = t
		}
	}

	out := make([]ANEDraftMILSlot, 0, len(eagle3MILSlotSpec))
	for _, spec := range eagle3MILSlotSpec {
		slot := spec
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

// ProbeANEDraftMILMap builds the Eagle3 tensor → MIL slot plan for a draft tag.
func ProbeANEDraftMILMap(_ context.Context, preferred string) (ANEDraftMILMapResult, error) {
	out := ANEDraftMILMapResult{
		Mode: "draft_mil_map",
		Note: "phase2_proxy uses sidecar extract or synthetic conv; phase3_subgraph awaits full tensor MIL compile",
	}

	if runtime.GOOS != "darwin" {
		out.Blockers = append(out.Blockers, "darwin only")
		return out, nil
	}

	entries, err := ListANEDraftInventory()
	if err != nil {
		return out, err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		out.Blockers = append(out.Blockers, "no ANE draft target in local inventory")
		return out, fmt.Errorf("no ANE draft target in local inventory")
	}

	out.Tag = entry.Tag
	out.SpecType = entry.SpecType
	out.ProxyChannels = entry.ProxyChannels
	out.ProxySpatial = entry.ProxySpatial
	out.WeightBlobBytes = draftMILWeightBlobBytes(entry.ProxyChannels)

	short := strings.SplitN(entry.Tag, ":", 2)[0]
	out.DraftSearchPaths = ANEDraftSidecarCandidates(short)

	draftPath, present := resolveDraftGGUFPath(entry)
	out.DraftGGUF = draftPath
	out.DraftSidecarPresent = present

	proxyReady := FindANEDraftDaemonBin() != ""
	var tensors []*ggml.Tensor
	if present && draftPath != "" {
		f, err := os.Open(draftPath)
		if err != nil {
			out.Blockers = append(out.Blockers, "open sidecar: "+err.Error())
		} else {
			meta, err := ggml.DecodeMetadata(f)
			_ = f.Close()
			if err != nil {
				out.Blockers = append(out.Blockers, "sidecar metadata: "+err.Error())
			} else {
				tensors = meta.Tensors().Items()
			}
		}
	} else {
		out.Blockers = append(out.Blockers, "eagle3 drafter GGUF missing")
		out.NextStep = "download eliza drafter: see scripts/setup_mtp_models.sh"
	}

	slots, matched, phase3Required := matchEagle3MILSlots(tensors, proxyReady)
	out.Slots = slots
	out.MatchedCount = matched
	out.RequiredCount = phase3Required + 1

	if present {
		phase3Matched := 0
		for _, s := range slots {
			if s.MILPhase == "phase3_subgraph" && s.Ready {
				phase3Matched++
			}
		}
		if phase3Matched < phase3Required {
			out.Blockers = append(out.Blockers, fmt.Sprintf("eagle3 tensors %d/%d matched for MIL extract", phase3Matched, phase3Required))
		}
	}

	if len(out.Blockers) == 0 {
		out.OK = true
		out.NextStep = "zerollama ane-draft-mil-extract --model " + short + " --out /tmp/ane-draft-weight.bin"
	} else if out.NextStep == "" {
		out.NextStep = "zerollama ane-draft-mil-status --model " + short
	}
	return out, nil
}

// RunANEDraftMILMapJSON writes draft MIL map JSON to w.
func RunANEDraftMILMapJSON(ctx context.Context, w io.Writer, preferred string) error {
	res, err := ProbeANEDraftMILMap(ctx, preferred)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
