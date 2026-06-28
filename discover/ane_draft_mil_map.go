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

// ANEDraftMILSlot maps a draft sidecar GGUF tensor role to a future ANE MIL weight slot.
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

// ANEDraftMILMapResult is the sidecar tensor → MIL compile plan.
type ANEDraftMILMapResult struct {
	OK                  bool               `json:"ok"`
	Mode                string             `json:"mode"`
	Tag                 string             `json:"tag,omitempty"`
	SpecType            string             `json:"spec_type,omitempty"`
	SidecarArchitecture string             `json:"sidecar_architecture,omitempty"`
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

// dflashMILSlotSpec maps dflash-draft sidecar tensors (llama.cpp dflash-draft.cpp).
// Why phase2_proxy vs phase3_subgraph: lab conv extract proves ANE latency today; full
// dflash_fc + attn slots await B7 MIL compile (see docs/ane-draft-inprocess.md).
var dflashMILSlotSpec = []ANEDraftMILSlot{
	{
		Slot:           "dflash_fc",
		Role:           "fuse target layer hiddens → draft hidden (build_lora_mm dflash_fc)",
		TensorPatterns: []string{"dflash_fc.weight"},
		MILPhase:       "phase3_subgraph",
		Note:           "matmul [n_target_features × n_embd]; not the lab conv proxy",
	},
	{
		Slot:           "dflash_hidden_norm",
		Role:           "RMS norm on fused target hidden",
		TensorPatterns: []string{"dflash_hidden_norm.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "decoder_attn_q",
		Role:           "draft decoder attention Q projection",
		TensorPatterns: []string{"blk.0.attn_q.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "decoder_ffn_gate",
		Role:           "draft decoder FFN gate (lab proxy extract source)",
		TensorPatterns: []string{"blk.0.ffn_gate.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "proxy_conv_w0",
		Role:           "lab ANE conv proxy (latency stand-in for full draft step)",
		TensorPatterns: []string{},
		MILPhase:       "phase2_proxy",
		Note:           "top-left slice of blk.0.ffn_gate.weight via ane-draft-mil-extract",
	},
}

// qwen35DrafterMILSlotSpec maps full qwen35 drafter GGUFs used as eliza dflash sidecars.
var qwen35DrafterMILSlotSpec = []ANEDraftMILSlot{
	{
		Slot:           "decoder_ffn_gate",
		Role:           "draft block0 FFN gate (conv proxy extract source)",
		TensorPatterns: []string{"blk.0.ffn_gate.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "decoder_ffn_norm",
		Role:           "draft block0 FFN RMS norm gamma (B3 mul-after-conv)",
		TensorPatterns: []string{"blk.0.ffn_norm.weight", "blk.0.attn_norm.weight"},
		MILPhase:       "phase3_subgraph",
	},
	{
		Slot:           "proxy_conv_w0",
		Role:           "lab ANE conv proxy (latency stand-in for full draft step)",
		TensorPatterns: []string{},
		MILPhase:       "phase2_proxy",
		Note:           "sidecar extract + optional gamma via ane-draft-mil-bundle",
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

// ProbeANEDraftMILMap builds the sidecar tensor → MIL slot plan for a draft tag.
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
		out.Blockers = append(out.Blockers, "draft sidecar GGUF missing")
		out.NextStep = "download eliza drafter: see scripts/setup_mtp_models.sh"
	}

	sidecarArch := ""
	if present && draftPath != "" {
		if arch, err := ProbeSidecarArchitecture(draftPath); err == nil {
			sidecarArch = arch
			out.SidecarArchitecture = arch
		}
	}

	slotSpec := milSlotSpecForDraft(entry.SpecType, sidecarArch)
	slots, matched, phase3Required := matchMILSlots(slotSpec, tensors, proxyReady)
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
			label := sidecarArch
			if label == "" {
				label = "draft"
			}
			out.Blockers = append(out.Blockers, fmt.Sprintf("%s tensors %d/%d matched for MIL extract", label, phase3Matched, phase3Required))
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
