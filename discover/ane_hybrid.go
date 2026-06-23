package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// ANEHybridLabResult is Metal→IOSurface→ANE + draft conv at GGUF proxy dimensions.
type ANEHybridLabResult struct {
	OK                  bool                  `json:"ok"`
	Model               string                `json:"model,omitempty"`
	Tag                 string                `json:"tag,omitempty"`
	SpecType            string                `json:"spec_type,omitempty"`
	ProxyChannels       int                   `json:"proxy_channels"`
	ProxySpatial        int                   `json:"proxy_spatial"`
	DraftSidecarPresent bool                  `json:"draft_sidecar_present"`
	ANEDraftEnv         bool                  `json:"ane_draft_env"`
	MetalHandoff        ANEMetalHandoffResult `json:"metal_handoff"`
	DraftBench          ANEDraftBenchResult   `json:"draft_bench"`
	Note                string                `json:"note,omitempty"`
	Error               string                `json:"error,omitempty"`
}

func hybridFromProxy(proxy ANEModelProxyDims) ANEDraftEntry {
	return ANEDraftEntry{
		Name:                proxy.Name,
		Tag:                 proxy.Tag,
		SpecType:            proxy.SpecType,
		EmbeddingLength:     proxy.EmbeddingLength,
		ProxyChannels:       proxy.ProxyChannels,
		ProxySpatial:        proxy.ProxySpatial,
		DraftSidecarPresent: proxy.DraftSidecarPresent,
	}
}

// ProbeANEHybridLabForModel runs hybrid lab smokes at GGUF-derived proxy dimensions.
func ProbeANEHybridLabForModel(ctx context.Context, preferred string, quick bool) (ANEHybridLabResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEHybridLabResult{}, fmt.Errorf("ane hybrid lab: darwin only")
	}

	proxy, err := ResolveANEModelProxyDims(preferred)
	if err != nil {
		return ANEHybridLabResult{}, err
	}
	entry := hybridFromProxy(proxy)
	ch, sp := proxy.ProxyChannels, proxy.ProxySpatial
	metal, err := ProbeANEMetalHandoffDims(ctx, ch, sp, quick)
	if err != nil {
		return ANEHybridLabResult{
			OK:            false,
			Model:         entry.Name,
			Tag:           entry.Tag,
			SpecType:      entry.SpecType,
			ProxyChannels: ch,
			ProxySpatial:  sp,
			ANEDraftEnv:   envconfig.ANEDraftEnabled(),
			Error:         err.Error(),
		}, err
	}

	draft, derr := ProbeANEDraftBenchDims(ctx, ch, sp, quick)
	res := ANEHybridLabResult{
		OK:                  true,
		Model:               entry.Name,
		Tag:                 entry.Tag,
		SpecType:            entry.SpecType,
		ProxyChannels:       ch,
		ProxySpatial:        sp,
		DraftSidecarPresent: entry.DraftSidecarPresent,
		ANEDraftEnv:         envconfig.ANEDraftEnabled(),
		MetalHandoff:        metal,
		DraftBench:          draft,
		Note:                "Metal prefill → IOSurface → ANE draft conv proxy; ggml hook + eagle3 weights are follow-on",
	}
	if derr != nil {
		res.OK = false
		res.Error = derr.Error()
		return res, derr
	}
	return res, nil
}

// ANEDraftLabEnabled reports whether hybrid ANE draft lab routing is opted in.
func ANEDraftLabEnabled() bool {
	return envconfig.ANEDraftEnabled()
}

// RunANEHybridLabForModel writes hybrid lab JSON to w.
func RunANEHybridLabForModel(ctx context.Context, w io.Writer, preferred string, quick bool) error {
	res, err := ProbeANEHybridLabForModel(ctx, preferred, quick)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
