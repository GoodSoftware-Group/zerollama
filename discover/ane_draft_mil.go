package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
)

const sidecarTensorSampleMax = 12

// ANEDraftMILStatus reports draft sidecar → MIL compile readiness and blockers.
type ANEDraftMILStatus struct {
	OK                   bool                    `json:"ok"`
	Mode                 string                  `json:"mode"`
	Tag                  string                  `json:"tag,omitempty"`
	SpecType             string                  `json:"spec_type,omitempty"`
	SidecarArchitecture  string                  `json:"sidecar_architecture,omitempty"`
	DraftSidecarPresent  bool                    `json:"draft_sidecar_present"`
	DraftGGUF            string                  `json:"draft_gguf,omitempty"`
	BaseGGUF             string                  `json:"base_gguf,omitempty"`
	ProxyChannels        int                     `json:"proxy_channels"`
	ProxySpatial         int                     `json:"proxy_spatial"`
	SidecarTensorCount   int                     `json:"sidecar_tensor_count"`
	SidecarSampleTensors []string                `json:"sidecar_sample_tensors,omitempty"`
	GGMLHook             GGMLIOSurfaceHookStatus `json:"ggml_hook"`
	ANEDraftEnv          bool                    `json:"ane_draft_env"`
	LabBinsReady         bool                    `json:"lab_bins_ready"`
	Blockers             []string                `json:"blockers"`
	NextStep             string                  `json:"next_step,omitempty"`
	Note                 string                  `json:"note,omitempty"`
}

// ProbeANEDraftSidecarTensors reads tensor headers from a draft sidecar GGUF.
func ProbeANEDraftSidecarTensors(path string) (count int, sample []string, err error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return 0, nil, fmt.Errorf("empty sidecar path")
	}
	f, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer f.Close()

	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return 0, nil, err
	}
	items := meta.Tensors().Items()
	count = len(items)
	sample = make([]string, 0, sidecarTensorSampleMax)
	for i, t := range items {
		if i >= sidecarTensorSampleMax {
			break
		}
		if t != nil && t.Name != "" {
			sample = append(sample, t.Name)
		}
	}
	return count, sample, nil
}

// ProbeANEDraftMILStatus inspects sidecar/MIL readiness for a draft model tag.
func ProbeANEDraftMILStatus(_ context.Context, preferred string) (ANEDraftMILStatus, error) {
	out := ANEDraftMILStatus{
		Mode:        "draft_mil_status",
		GGMLHook:    ProbeGGMLIOSurfaceHookStatus(),
		ANEDraftEnv: envconfig.ANEDraftEnabled(),
		LabBinsReady: FindANEDraftDaemonBin() != "" &&
			FindANEGGMLMapSmokeBin() != "" &&
			FindANEProbeBin() != "",
		Note: "sidecar tensor extract via ane-draft-mil-extract; phase3 subgraph MIL is follow-on",
	}

	if runtime.GOOS != "darwin" {
		out.Blockers = append(out.Blockers, "darwin only")
		out.NextStep = "run on Apple Silicon"
		return out, nil
	}

	entries, err := ListANEDraftInventory()
	if err != nil {
		return out, err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		out.Blockers = append(out.Blockers, "no ANE draft target in local inventory")
		out.NextStep = "pull eliza-*-dflash or set spec_type draft-eagle3"
		return out, fmt.Errorf("no ANE draft target in local inventory")
	}

	draftPath, draftPresent := resolveDraftGGUFPath(entry)
	entry.DraftGGUF = draftPath
	entry.DraftSidecarPresent = draftPresent

	out.Tag = entry.Tag
	out.SpecType = entry.SpecType
	out.DraftSidecarPresent = entry.DraftSidecarPresent
	out.DraftGGUF = entry.DraftGGUF
	out.BaseGGUF = entry.BaseGGUF
	out.ProxyChannels = entry.ProxyChannels
	out.ProxySpatial = entry.ProxySpatial

	if !out.LabBinsReady {
		out.Blockers = append(out.Blockers, "ANE lab binaries missing")
	}
	if !out.GGMLHook.APIAvailable {
		out.Blockers = append(out.Blockers, "ggml IOSurface hook not built")
	}
	if !entry.DraftSidecarPresent {
		short := strings.SplitN(entry.Tag, ":", 2)[0]
		out.Blockers = append(out.Blockers, "draft sidecar GGUF missing")
		out.NextStep = "download drafter (scripts/setup_mtp_models.sh) or place at " + strings.Join(ANEDraftSidecarCandidates(short), " | ")
	} else if entry.DraftGGUF != "" {
		if arch, err := ProbeSidecarArchitecture(entry.DraftGGUF); err == nil {
			out.SidecarArchitecture = arch
		}
		count, sample, err := ProbeANEDraftSidecarTensors(entry.DraftGGUF)
		if err != nil {
			out.Blockers = append(out.Blockers, "sidecar tensor inspect: "+err.Error())
		} else {
			out.SidecarTensorCount = count
			out.SidecarSampleTensors = sample
			if count == 0 {
				out.Blockers = append(out.Blockers, "sidecar GGUF has no tensors")
			}
		}
	}

	if len(out.Blockers) == 0 {
		out.OK = true
		out.NextStep = "zerollama ane-draft-mil-extract --model " + strings.SplitN(entry.Tag, ":", 2)[0]
	} else if out.NextStep == "" {
		out.NextStep = "./scripts/ane_probe_build.sh; place sidecar; ZEROLLAMA_ANE_DRAFT=1 ane-draft-router-smoke"
	}
	return out, nil
}

// RunANEDraftMILStatusJSON writes draft MIL status JSON to w.
func RunANEDraftMILStatusJSON(ctx context.Context, w io.Writer, preferred string) error {
	st, err := ProbeANEDraftMILStatus(ctx, preferred)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(st)
		return err
	}
	return enc.Encode(st)
}

func draftMILBlockers(sidecarPresent, labBins, ggmlHook bool) []string {
	var blockers []string
	if !labBins {
		blockers = append(blockers, "ANE lab binaries missing")
	}
	if !ggmlHook {
		blockers = append(blockers, "ggml IOSurface hook not built")
	}
	if !sidecarPresent {
		blockers = append(blockers, "draft sidecar GGUF missing")
	}
	return blockers
}
