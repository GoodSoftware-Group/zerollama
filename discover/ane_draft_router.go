package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"sync"
)

// ANEDraftRouter manages a long-lived ane-draft-daemon session for repeated draft steps.
// Lab-only today; future scheduler hook when ZEROLLAMA_ANE_DRAFT=1 and ggml map lands in-process.
type ANEDraftRouter struct {
	mu      sync.Mutex
	sess    *draftDaemonSession
	ready   ANEDraftDaemonReady
	tag     string
	ch      int
	sp      int
	steps   int
	started bool
}

// ANEDraftStepResult is one map_fill + eval_ane cycle on a warm daemon session.
type ANEDraftStepResult struct {
	Step    int                  `json:"step"`
	MapFill ANEGGMLMapFillResult `json:"map_fill"`
	Eval    ANEDraftDaemonBench  `json:"eval"`
}

// ANEDraftRouterSmokeResult reports multi-step router throughput.
type ANEDraftRouterSmokeResult struct {
	OK            bool                  `json:"ok"`
	Mode          string                `json:"mode"`
	Tag           string                `json:"tag,omitempty"`
	ProxyChannels int                   `json:"proxy_channels"`
	ProxySpatial  int                   `json:"proxy_spatial"`
	Ready         ANEDraftDaemonReady   `json:"ready"`
	Steps         []ANEDraftStepResult  `json:"steps"`
	AvgEvalMS     float64               `json:"avg_eval_ms"`
	AvgMapFillMS  float64               `json:"avg_map_fill_ms"`
	KernelReused  bool                  `json:"kernel_reused"`
	Note          string                `json:"note,omitempty"`
	Error         string                `json:"error,omitempty"`
}

// NewANEDraftRouter returns an idle router (call Start before DraftStep).
func NewANEDraftRouter() *ANEDraftRouter {
	return &ANEDraftRouter{}
}

// Start compiles the draft kernel once and keeps the daemon session open.
func (r *ANEDraftRouter) Start(ctx context.Context, preferred string) (ANEDraftDaemonReady, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftDaemonReady{}, fmt.Errorf("ane draft router: darwin only")
	}
	if !ANEDraftLabEnabled() {
		return ANEDraftDaemonReady{}, fmt.Errorf("ane draft router: ZEROLLAMA_ANE_DRAFT not enabled")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.started {
		return r.ready, nil
	}

	ch, sp := 64, 16
	tag := ""
	if preferred != "" {
		proxy, err := ResolveANEModelProxyDims(preferred)
		if err != nil {
			return ANEDraftDaemonReady{}, err
		}
		ch, sp = proxy.ProxyChannels, proxy.ProxySpatial
		tag = proxy.Tag
	}

	sess, ready, err := startDraftDaemonSessionLongLived(ctx, ch, sp)
	if err != nil {
		return ready, err
	}

	r.sess = sess
	r.ready = ready
	r.tag = tag
	r.ch = ch
	r.sp = sp
	r.started = true
	return ready, nil
}

// DraftStep runs ggml map fill then ANE eval on the warm session.
func (r *ANEDraftRouter) DraftStep(ctx context.Context, fill float64) (ANEDraftStepResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.started || r.sess == nil {
		return ANEDraftStepResult{}, fmt.Errorf("ane draft router: not started")
	}
	if fill == 0 {
		fill = 0.01
	}

	mapFill, err := r.sess.sendMapFill(ctx, fill)
	if err != nil {
		return ANEDraftStepResult{}, err
	}
	eval, err := r.sess.sendCommand(ctx, map[string]any{"cmd": "eval_ane"})
	if err != nil {
		return ANEDraftStepResult{MapFill: mapFill, Eval: eval}, err
	}

	r.steps++
	return ANEDraftStepResult{
		Step:    r.steps,
		MapFill: mapFill,
		Eval:    eval,
	}, nil
}

// Close shuts down the daemon session.
func (r *ANEDraftRouter) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.started || r.sess == nil {
		return nil
	}
	err := r.sess.close()
	r.sess = nil
	r.started = false
	r.steps = 0
	return err
}

// Ready reports whether the router holds a live daemon session.
func (r *ANEDraftRouter) Ready() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.started && r.sess != nil
}

// Status returns a snapshot for diagnostics.
func (r *ANEDraftRouter) Status() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	return map[string]any{
		"started":        r.started,
		"tag":            r.tag,
		"proxy_channels": r.ch,
		"proxy_spatial":  r.sp,
		"steps":          r.steps,
		"surface_id":     r.ready.SurfaceID,
		"compile_count":  r.ready.CompileCount,
	}
}

// ProbeANEDraftRouterSmoke runs N draft steps on one router session.
func ProbeANEDraftRouterSmoke(ctx context.Context, preferred string, steps int, quick bool) (ANEDraftRouterSmokeResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftRouterSmokeResult{}, fmt.Errorf("draft router smoke: darwin only")
	}
	if !ANEDraftLabEnabled() {
		return ANEDraftRouterSmokeResult{}, fmt.Errorf("draft router smoke: set ZEROLLAMA_ANE_DRAFT=1")
	}
	if steps <= 0 {
		steps = defaultRouterSteps(quick)
	}

	router := NewANEDraftRouter()
	defer func() { _ = router.Close() }()

	ready, err := router.Start(ctx, preferred)
	out := ANEDraftRouterSmokeResult{
		Mode:          "draft_router_smoke",
		ProxyChannels: router.ch,
		ProxySpatial:  router.sp,
		Tag:           router.tag,
		Ready:         ready,
		Note:          "scheduler hook prototype — long-lived daemon, map_fill + eval_ane per draft step",
	}
	if err != nil {
		out.OK = false
		out.Error = err.Error()
		return out, err
	}
	out.ProxyChannels = router.ch
	out.ProxySpatial = router.sp
	out.Tag = router.tag

	var evalSum, mapSum float64
	for i := 0; i < steps; i++ {
		step, err := router.DraftStep(ctx, 0.01)
		if err != nil {
			out.Steps = append(out.Steps, step)
			out.OK = false
			out.Error = err.Error()
			return out, err
		}
		out.Steps = append(out.Steps, step)
		evalSum += step.Eval.EvalMS
		mapSum += step.MapFill.MetalFillMS
	}

	out.AvgEvalMS = evalSum / float64(steps)
	out.AvgMapFillMS = mapSum / float64(steps)
	out.KernelReused = ready.CompileCount == 1
	for _, s := range out.Steps {
		if s.Eval.CompileCount != 1 || s.MapFill.CompileCount != 1 {
			out.KernelReused = false
			break
		}
	}
	out.OK = out.KernelReused
	if !out.KernelReused {
		out.Error = "compile_count drift across steps"
		return out, fmt.Errorf("%s", out.Error)
	}
	return out, nil
}

// RunANEDraftRouterSmokeJSON writes router smoke JSON to w.
func RunANEDraftRouterSmokeJSON(ctx context.Context, w io.Writer, preferred string, steps int, quick bool) error {
	res, err := ProbeANEDraftRouterSmoke(ctx, preferred, steps, quick)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

func defaultRouterSteps(quick bool) int {
	if quick {
		return 3
	}
	return 5
}
