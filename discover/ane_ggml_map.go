package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
)

// ANEGGMLMapSmokeResult is parent ggml_metal_buffer_map fill + daemon ANE eval.
type ANEGGMLMapSmokeResult struct {
	OK            bool                 `json:"ok"`
	Mode          string               `json:"mode"`
	Tag           string               `json:"tag,omitempty"`
	ProxyChannels int                  `json:"proxy_channels"`
	ProxySpatial  int                  `json:"proxy_spatial"`
	Ready         ANEDraftDaemonReady  `json:"ready"`
	MapFill       ANEGGMLMapFillResult `json:"map_fill"`
	Eval          ANEDraftDaemonBench  `json:"eval"`
	Note          string               `json:"note,omitempty"`
	Error         string               `json:"error,omitempty"`
}

// ANEGGMLMapFillResult is JSON from daemon map_fill or ane-ggml-map-smoke (same-process).
type ANEGGMLMapFillResult struct {
	OK               bool    `json:"ok"`
	Mode             string  `json:"mode"`
	Event            string  `json:"event,omitempty"`
	SurfaceID        uint32  `json:"surface_id"`
	SurfaceBytes     int     `json:"surface_bytes"`
	MappedBytes      int     `json:"mapped_bytes"`
	MappedPageOffset int     `json:"mapped_page_offset"`
	TensorOffset     int     `json:"tensor_offset"`
	MetalFillMS      float64 `json:"metal_fill_ms"`
	GGMLMapOK        bool    `json:"ggml_map_ok"`
	CompileCount     int     `json:"compile_count,omitempty"`
	Source           string  `json:"source"`
	Note             string  `json:"note,omitempty"`
	Error            string  `json:"error,omitempty"`
}

// FindANEGGMLMapSmokeBin locates the ggml map smoke binary (same-process reference only).
func FindANEGGMLMapSmokeBin() string {
	return aneToolBin("ane-ggml-map-smoke")
}

func parseMapFillJSON(line []byte) (ANEGGMLMapFillResult, error) {
	var res ANEGGMLMapFillResult
	if jerr := json.Unmarshal(line, &res); jerr != nil {
		return ANEGGMLMapFillResult{}, fmt.Errorf("map_fill json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "map_fill returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// ProbeANEGGMLMapSmoke runs daemon compile-once + in-process ggml map fill + ANE eval.
func ProbeANEGGMLMapSmoke(ctx context.Context, preferred string, quick bool) (ANEGGMLMapSmokeResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEGGMLMapSmokeResult{}, fmt.Errorf("ggml map smoke: darwin only")
	}

	ch, sp := 64, 16
	tag := ""
	if preferred != "" {
		proxy, err := ResolveANEModelProxyDims(preferred)
		if err != nil {
			return ANEGGMLMapSmokeResult{}, err
		}
		ch, sp = proxy.ProxyChannels, proxy.ProxySpatial
		tag = proxy.Tag
	}

	sess, ready, err := startDraftDaemonSession(ctx, ch, sp)
	if err != nil {
		out := ANEGGMLMapSmokeResult{
			OK: false, Mode: "ggml_map_handoff", Tag: tag,
			ProxyChannels: ch, ProxySpatial: sp, Ready: ready, Error: err.Error(),
		}
		return out, err
	}
	defer func() { _ = sess.close() }()

	mapFill, err := sess.sendMapFill(ctx, 0.01)
	if err != nil {
		out := ANEGGMLMapSmokeResult{
			OK: false, Mode: "ggml_map_handoff", Tag: tag,
			ProxyChannels: ch, ProxySpatial: sp, Ready: ready, MapFill: mapFill, Error: err.Error(),
		}
		return out, err
	}

	eval, err := sess.sendCommand(ctx, map[string]any{"cmd": "eval_ane"})
	if err != nil {
		out := ANEGGMLMapSmokeResult{
			OK: false, Mode: "ggml_map_handoff", Tag: tag,
			ProxyChannels: ch, ProxySpatial: sp, Ready: ready, MapFill: mapFill, Eval: eval, Error: err.Error(),
		}
		return out, err
	}

	return ANEGGMLMapSmokeResult{
		OK:            mapFill.GGMLMapOK && eval.OK,
		Mode:          "ggml_map_handoff",
		Tag:           tag,
		ProxyChannels: ch,
		ProxySpatial:  sp,
		Ready:         ready,
		MapFill:       mapFill,
		Eval:          eval,
		Note:          "ggml_metal_buffer_map-equivalent fill in daemon process (ANE IOSurface is same-process); production ggml hook is in-process",
	}, nil
}

// RunANEGGMLMapSmokeJSON writes ggml map handoff JSON to w.
func RunANEGGMLMapSmokeJSON(ctx context.Context, w io.Writer, preferred string, quick bool) error {
	res, err := ProbeANEGGMLMapSmoke(ctx, preferred, quick)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
