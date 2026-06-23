package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
)

// ANEDraftSurfaceHandoffResult is Metal→IOSurface→ANE draft conv at model proxy dims.
type ANEDraftSurfaceHandoffResult struct {
	OK                  bool                  `json:"ok"`
	Mode                string                `json:"mode"`
	Tag                 string                `json:"tag,omitempty"`
	EmbeddingLength     int                   `json:"embedding_length,omitempty"`
	ProxyChannels       int                   `json:"proxy_channels"`
	ProxySpatial        int                   `json:"proxy_spatial"`
	SpecType            string                `json:"spec_type,omitempty"`
	DraftSidecarPresent bool                  `json:"draft_sidecar_present,omitempty"`
	ProxySource         string                `json:"proxy_source,omitempty"`
	Handoff             ANEMetalHandoffResult `json:"handoff"`
	Note                string                `json:"note,omitempty"`
	Error               string                `json:"error,omitempty"`
}

// ProbeANEDraftSurfaceHandoffForModel runs draft-shaped Metal IOSurface handoff at GGUF proxy dims.
func ProbeANEDraftSurfaceHandoffForModel(ctx context.Context, preferred string, quick bool) (ANEDraftSurfaceHandoffResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftSurfaceHandoffResult{}, fmt.Errorf("draft surface handoff: darwin only")
	}
	proxy, err := ResolveANEModelProxyDims(preferred)
	if err != nil {
		return ANEDraftSurfaceHandoffResult{}, err
	}
	handoff, err := ProbeANEMetalHandoffDims(ctx, proxy.ProxyChannels, proxy.ProxySpatial, quick)
	out := ANEDraftSurfaceHandoffResult{
		OK:                  err == nil,
		Mode:                "draft_surface_handoff",
		Tag:                 proxy.Tag,
		EmbeddingLength:     proxy.EmbeddingLength,
		ProxyChannels:       proxy.ProxyChannels,
		ProxySpatial:        proxy.ProxySpatial,
		SpecType:            proxy.SpecType,
		DraftSidecarPresent: proxy.DraftSidecarPresent,
		ProxySource:         proxy.Source,
		Handoff:             handoff,
		Note:                "Metal producer fills IOSurface; ANE draft conv eval — surface_id is ggml handoff export target",
	}
	if err != nil {
		out.Error = err.Error()
		return out, err
	}
	return out, nil
}

// RunANEDraftSurfaceHandoffForModel writes draft surface handoff JSON to w.
func RunANEDraftSurfaceHandoffForModel(ctx context.Context, w io.Writer, preferred string, quick bool) error {
	res, err := ProbeANEDraftSurfaceHandoffForModel(ctx, preferred, quick)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
