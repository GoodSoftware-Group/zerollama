package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
)

// ANEInprocessStepResult is one ggml map_fill + eval_ane cycle in the same process as the ANE kernel.
type ANEInprocessStepResult struct {
	Step    int                    `json:"step"`
	MapFill map[string]interface{} `json:"map_fill"`
	Eval    map[string]interface{} `json:"eval"`
}

// ANEInprocessSmokeResult reports same-process draft-step throughput (B1 integration contract).
type ANEInprocessSmokeResult struct {
	OK            bool                     `json:"ok"`
	Mode          string                   `json:"mode"`
	Tag           string                   `json:"tag,omitempty"`
	ProxyChannels int                      `json:"proxy_channels"`
	ProxySpatial  int                      `json:"proxy_spatial"`
	SurfaceID     uint32                   `json:"surface_id,omitempty"`
	WeightFile    string                   `json:"weight_file,omitempty"`
	WeightSource  string                   `json:"weight_source,omitempty"`
	CompileMS     float64                  `json:"compile_ms,omitempty"`
	CompileCount  int                      `json:"compile_count,omitempty"`
	KernelReused  bool                     `json:"kernel_reused"`
	AvgEvalMS     float64                  `json:"avg_eval_ms"`
	AvgMapFillMS  float64                  `json:"avg_map_fill_ms"`
	ExportEnv     map[string]string        `json:"export_env,omitempty"`
	Steps         []ANEInprocessStepResult `json:"steps,omitempty"`
	Note          string                   `json:"note,omitempty"`
	Error         string                   `json:"error,omitempty"`
}

// FindANEInprocessSmokeBin locates the same-process ANE draft smoke binary.
func FindANEInprocessSmokeBin() string {
	return aneToolBin("ane-inprocess-smoke")
}

// ProbeANEInprocessSmoke runs compile-once + N ggml map + ANE eval steps in one OS process.
func ProbeANEInprocessSmoke(ctx context.Context, preferred string, steps int, quick bool) (ANEInprocessSmokeResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEInprocessSmokeResult{}, fmt.Errorf("ane inprocess smoke: darwin only")
	}
	bin := FindANEInprocessSmokeBin()
	if bin == "" {
		return ANEInprocessSmokeResult{}, fmt.Errorf("ane-inprocess-smoke not found — run ./scripts/ane/ane_probe_build.sh")
	}

	ch, sp := 64, 16
	tag := ""
	weightFile := ""
	if preferred != "" {
		proxy, err := ResolveANEModelProxyDims(preferred)
		if err != nil {
			return ANEInprocessSmokeResult{}, err
		}
		ch, sp = proxy.ProxyChannels, proxy.ProxySpatial
		tag = proxy.Tag

		if entries, err := ListANEDraftInventory(); err == nil {
			if entry, ok := SelectANEDraftModel(entries, preferred); ok {
				draftPath, sidecarPresent := resolveDraftGGUFPath(entry)
				if sidecarPresent {
					entry.DraftGGUF = draftPath
					entry.DraftSidecarPresent = true
					if path, _, err := MaterializeANEDraftWeightFile(entry, ""); err == nil {
						weightFile = path
					}
				}
			}
		}
	}

	if steps <= 0 {
		if quick {
			steps = 3
		} else {
			steps = 5
		}
	}

	args := []string{
		"--channels", fmt.Sprintf("%d", ch),
		"--spatial", fmt.Sprintf("%d", sp),
		"--steps", fmt.Sprintf("%d", steps),
	}
	if weightFile != "" {
		args = append(args, "--weight-file", weightFile)
	}
	if quick {
		args = append(args, "--quick")
	}

	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEInprocessSmokeResult{}, err
	}

	var raw map[string]interface{}
	if jerr := json.Unmarshal(out, &raw); jerr != nil {
		return ANEInprocessSmokeResult{}, fmt.Errorf("ane-inprocess-smoke json: %w", jerr)
	}

	res := ANEInprocessSmokeResult{
		Mode:          "ane_inprocess_smoke",
		Tag:           tag,
		ProxyChannels: ch,
		ProxySpatial:  sp,
		WeightFile:    weightFile,
		Note:          "same-process IOSurface owner + ggml map fill — prerequisite for llama-server draft hook",
	}
	if v, ok := raw["ok"].(bool); ok {
		res.OK = v
	}
	if v, ok := raw["surface_id"].(float64); ok {
		res.SurfaceID = uint32(v)
	}
	if v, ok := raw["compile_ms"].(float64); ok {
		res.CompileMS = v
	}
	if v, ok := raw["compile_count"].(float64); ok {
		res.CompileCount = int(v)
	}
	if v, ok := raw["kernel_reused"].(bool); ok {
		res.KernelReused = v
	}
	if v, ok := raw["avg_eval_ms"].(float64); ok {
		res.AvgEvalMS = v
	}
	if v, ok := raw["avg_map_fill_ms"].(float64); ok {
		res.AvgMapFillMS = v
	}
	if v, ok := raw["weight_source"].(string); ok {
		res.WeightSource = v
	}
	if v, ok := raw["error"].(string); ok {
		res.Error = v
	}
	if stepsRaw, ok := raw["steps"].([]interface{}); ok {
		for _, item := range stepsRaw {
			stepMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			step := ANEInprocessStepResult{}
			if n, ok := stepMap["step"].(float64); ok {
				step.Step = int(n)
			}
			if mf, ok := stepMap["map_fill"].(map[string]interface{}); ok {
				step.MapFill = mf
			}
			if ev, ok := stepMap["eval"].(map[string]interface{}); ok {
				step.Eval = ev
			}
			res.Steps = append(res.Steps, step)
		}
	}

	if err != nil {
		if res.Error == "" {
			res.Error = err.Error()
		}
		return res, err
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "ane-inprocess-smoke returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	if res.SurfaceID != 0 && res.ProxyChannels > 0 {
		bytes := res.ProxyChannels * res.ProxySpatial * 4
		if v, ok := raw["surface_bytes"].(float64); ok && v > 0 {
			bytes = int(v)
		}
		res.ExportEnv = map[string]string{
			"ZEROLLAMA_ANE_DRAFT":               "1",
			"ZEROLLAMA_ANE_DRAFT_CHANNELS":      fmt.Sprintf("%d", res.ProxyChannels),
			"ZEROLLAMA_ANE_DRAFT_SPATIAL":       fmt.Sprintf("%d", res.ProxySpatial),
			"ZEROLLAMA_ANE_DRAFT_SURFACE_ID":    fmt.Sprintf("%d", res.SurfaceID),
			"ZEROLLAMA_ANE_DRAFT_SURFACE_BYTES": fmt.Sprintf("%d", bytes),
		}
		if preferred != "" {
			if entries, err := ListANEDraftInventory(); err == nil {
				if entry, ok := SelectANEDraftModel(entries, preferred); ok {
					draftPath, present := resolveDraftGGUFPath(entry)
					if present && draftPath != "" {
						if manifest, _, err := MaterializeANEDraftWeightBundle(entry); err == nil {
							ch := entry.ProxyChannels
							if ch <= 0 {
								ch, _ = DraftANEProxyDims(entry.EmbeddingLength)
							}
							manifestPath := aneDraftWeightManifestPath(draftPath, ch)
							for k, v := range ExportEnvForManifest(manifest, manifestPath) {
								res.ExportEnv[k] = v
							}
						}
					}
				}
			}
		}
		if res.WeightFile != "" && res.ExportEnv["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE"] == "" {
			res.ExportEnv["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE"] = res.WeightFile
		}
	}
	return res, nil
}

// RunANEInprocessSmokeJSON writes in-process smoke JSON to w.
func RunANEInprocessSmokeJSON(ctx context.Context, w io.Writer, preferred string, steps int, quick bool) error {
	res, err := ProbeANEInprocessSmoke(ctx, preferred, steps, quick)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
