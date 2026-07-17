package discover

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const aneDraftDaemonTimeout = 180 * time.Second

// ANEDraftDaemonReady is the first JSON line from ane-draft-daemon.
type ANEDraftDaemonReady struct {
	OK            bool    `json:"ok"`
	Event         string  `json:"event"`
	Mode          string  `json:"mode"`
	Channels      int     `json:"channels"`
	Spatial       int     `json:"spatial"`
	SurfaceID     uint32  `json:"surface_id"`
	SurfaceBytes  int     `json:"surface_bytes"`
	CompileMS     float64 `json:"compile_ms"`
	CompileCount  int     `json:"compile_count"`
	WeightSource  string  `json:"weight_source,omitempty"`
	Source        string  `json:"source"`
	Note          string  `json:"note,omitempty"`
	Error         string  `json:"error,omitempty"`
}

// ANEDraftDaemonBench is JSON from bench/eval responses.
type ANEDraftDaemonBench struct {
	OK           bool    `json:"ok"`
	Event        string  `json:"event"`
	Channels     int     `json:"channels,omitempty"`
	Spatial      int     `json:"spatial,omitempty"`
	Iters        int     `json:"iters,omitempty"`
	MetalFillMS  float64 `json:"metal_fill_ms"`
	EvalMS       float64 `json:"eval_ms"`
	ReadMS       float64 `json:"read_ms"`
	TotalMS      float64 `json:"total_ms"`
	CompileMS    float64 `json:"compile_ms,omitempty"`
	CompileCount int     `json:"compile_count"`
	EvalCount    int     `json:"eval_count"`
	SurfaceID    uint32  `json:"surface_id"`
	Source       string  `json:"source,omitempty"`
	Note         string  `json:"note,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// ANEDraftDaemonSmokeResult compares compile-once session vs one-shot handoff.
type ANEDraftDaemonSmokeResult struct {
	OK              bool                    `json:"ok"`
	Mode            string                  `json:"mode"`
	Tag             string                  `json:"tag,omitempty"`
	ProxyChannels   int                     `json:"proxy_channels"`
	ProxySpatial    int                     `json:"proxy_spatial"`
	Ready           ANEDraftDaemonReady     `json:"ready"`
	FirstBench      ANEDraftDaemonBench     `json:"first_bench"`
	SecondBench     ANEDraftDaemonBench     `json:"second_bench"`
	OneShotHandoff  ANEMetalHandoffResult   `json:"one_shot_handoff"`
	KernelReused    bool                    `json:"kernel_reused"`
	Note            string                  `json:"note,omitempty"`
	Error           string                  `json:"error,omitempty"`
}

// FindANEDraftDaemonBin locates the persistent draft daemon binary.
func FindANEDraftDaemonBin() string {
	return aneToolBin("ane-draft-daemon")
}

type draftDaemonSession struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	reader *bufio.Reader
	cancel context.CancelFunc
}

func startDraftDaemonSession(ctx context.Context, channels, spatial int) (*draftDaemonSession, ANEDraftDaemonReady, error) {
	return startDraftDaemonSessionOpts(ctx, channels, spatial, false, "")
}

func startDraftDaemonSessionLongLived(ctx context.Context, channels, spatial int) (*draftDaemonSession, ANEDraftDaemonReady, error) {
	return startDraftDaemonSessionLongLivedWeight(ctx, channels, spatial, "")
}

func startDraftDaemonSessionLongLivedWeight(ctx context.Context, channels, spatial int, weightFile string) (*draftDaemonSession, ANEDraftDaemonReady, error) {
	return startDraftDaemonSessionOpts(ctx, channels, spatial, true, weightFile)
}

func startDraftDaemonSessionOpts(ctx context.Context, channels, spatial int, longLived bool, weightFile string) (*draftDaemonSession, ANEDraftDaemonReady, error) {
	if runtime.GOOS != "darwin" {
		return nil, ANEDraftDaemonReady{}, fmt.Errorf("draft daemon: darwin only")
	}
	bin := FindANEDraftDaemonBin()
	if bin == "" {
		return nil, ANEDraftDaemonReady{}, fmt.Errorf("ane-draft-daemon not found — run ./scripts/ane/ane_probe_build.sh")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	var runCtx context.Context
	var cancel context.CancelFunc
	if longLived {
		runCtx, cancel = context.WithCancel(ctx)
	} else {
		runCtx, cancel = context.WithTimeout(ctx, aneDraftDaemonTimeout)
	}

	args := []string{}
	if channels > 0 {
		args = append(args, "--channels", fmt.Sprintf("%d", channels))
	}
	if spatial > 0 {
		args = append(args, "--spatial", fmt.Sprintf("%d", spatial))
	}
	if strings.TrimSpace(weightFile) != "" {
		args = append(args, "--weight-file", weightFile)
	}

	cmd := exec.CommandContext(runCtx, bin, args...)
	cmd.Env = osEnviron()
	cmd.Dir = filepath.Dir(bin)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, ANEDraftDaemonReady{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		_ = stdin.Close()
		return nil, ANEDraftDaemonReady{}, err
	}
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		cancel()
		_ = stdin.Close()
		return nil, ANEDraftDaemonReady{}, err
	}

	reader := bufio.NewReader(stdout)
	line, err := reader.ReadString('\n')
	if err != nil {
		cancel()
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		return nil, ANEDraftDaemonReady{}, fmt.Errorf("daemon ready: %w", err)
	}

	var ready ANEDraftDaemonReady
	if jerr := json.Unmarshal([]byte(strings.TrimSpace(line)), &ready); jerr != nil {
		cancel()
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		return nil, ANEDraftDaemonReady{}, fmt.Errorf("daemon ready json: %w", jerr)
	}
	if !ready.OK || ready.Event != "ready" {
		msg := ready.Error
		if msg == "" {
			msg = "daemon ready failed"
		}
		cancel()
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		return nil, ready, fmt.Errorf("%s", msg)
	}

	return &draftDaemonSession{cmd: cmd, stdin: stdin, reader: reader, cancel: cancel}, ready, nil
}

func (s *draftDaemonSession) sendMapFill(ctx context.Context, fill float64) (ANEGGMLMapFillResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	_ = ctx

	payload, err := json.Marshal(map[string]any{"cmd": "map_fill", "fill": fill})
	if err != nil {
		return ANEGGMLMapFillResult{}, err
	}
	payload = append(payload, '\n')
	if _, err := s.stdin.Write(payload); err != nil {
		return ANEGGMLMapFillResult{}, err
	}

	line, err := s.reader.ReadString('\n')
	if err != nil {
		return ANEGGMLMapFillResult{}, err
	}
	return parseMapFillJSON([]byte(strings.TrimSpace(line)))
}

func (s *draftDaemonSession) sendCommand(ctx context.Context, req map[string]any) (ANEDraftDaemonBench, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	_ = ctx

	payload, err := json.Marshal(req)
	if err != nil {
		return ANEDraftDaemonBench{}, err
	}
	payload = append(payload, '\n')
	if _, err := s.stdin.Write(payload); err != nil {
		return ANEDraftDaemonBench{}, err
	}

	line, err := s.reader.ReadString('\n')
	if err != nil {
		return ANEDraftDaemonBench{}, err
	}

	var resp ANEDraftDaemonBench
	if jerr := json.Unmarshal([]byte(strings.TrimSpace(line)), &resp); jerr != nil {
		return ANEDraftDaemonBench{}, fmt.Errorf("daemon response json: %w", jerr)
	}
	if !resp.OK {
		msg := resp.Error
		if msg == "" {
			msg = "daemon command failed"
		}
		return resp, fmt.Errorf("%s", msg)
	}
	return resp, nil
}

func (s *draftDaemonSession) close() error {
	if s == nil {
		return nil
	}
	_, _ = s.sendCommand(context.Background(), map[string]any{"cmd": "quit"})
	_ = s.stdin.Close()
	err := s.cmd.Wait()
	if s.cancel != nil {
		s.cancel()
	}
	return err
}

func osEnviron() []string {
	return append([]string{}, os.Environ()...)
}

func draftDaemonBenchIters(quick bool) int {
	if quick {
		return 10
	}
	return 30
}

// ProbeANEDraftDaemonBench runs a one-shot --bench via the daemon binary.
func ProbeANEDraftDaemonBench(ctx context.Context, channels, spatial int, quick bool) (ANEDraftDaemonBench, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftDaemonBench{}, fmt.Errorf("draft daemon bench: darwin only")
	}
	bin := FindANEDraftDaemonBin()
	if bin == "" {
		return ANEDraftDaemonBench{}, fmt.Errorf("ane-draft-daemon not found — run ./scripts/ane/ane_probe_build.sh")
	}
	args := []string{"--bench", "--iters", fmt.Sprintf("%d", draftDaemonBenchIters(quick))}
	if channels > 0 {
		args = append(args, "--channels", fmt.Sprintf("%d", channels))
	}
	if spatial > 0 {
		args = append(args, "--spatial", fmt.Sprintf("%d", spatial))
	}
	if quick {
		args = append(args, "--quick")
	}
	out, err := runANETool(ctx, bin, args)
	if err != nil && len(out) == 0 {
		return ANEDraftDaemonBench{}, err
	}
	var res ANEDraftDaemonBench
	if jerr := json.Unmarshal(out, &res); jerr != nil {
		return ANEDraftDaemonBench{}, fmt.Errorf("ane-draft-daemon json: %w", jerr)
	}
	if !res.OK {
		msg := res.Error
		if msg == "" {
			msg = "draft daemon bench returned ok=false"
		}
		return res, fmt.Errorf("%s", msg)
	}
	return res, nil
}

// ProbeANEDraftDaemonSmoke runs two bench rounds in one daemon session and compares to one-shot handoff.
func ProbeANEDraftDaemonSmoke(ctx context.Context, preferred string, quick bool) (ANEDraftDaemonSmokeResult, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftDaemonSmokeResult{}, fmt.Errorf("draft daemon smoke: darwin only")
	}

	ch, sp := 64, 16
	tag := ""
	if preferred != "" {
		proxy, err := ResolveANEModelProxyDims(preferred)
		if err != nil {
			return ANEDraftDaemonSmokeResult{}, err
		}
		ch, sp = proxy.ProxyChannels, proxy.ProxySpatial
		tag = proxy.Tag
	}

	iters := draftDaemonBenchIters(quick)
	sess, ready, err := startDraftDaemonSession(ctx, ch, sp)
	if err != nil {
		out := ANEDraftDaemonSmokeResult{
			OK:            false,
			Mode:          "draft_daemon_smoke",
			Tag:           tag,
			ProxyChannels: ch,
			ProxySpatial:  sp,
			Ready:         ready,
			Error:         err.Error(),
		}
		return out, err
	}
	defer func() { _ = sess.close() }()

	first, err := sess.sendCommand(ctx, map[string]any{"cmd": "bench", "iters": iters})
	if err != nil {
		out := ANEDraftDaemonSmokeResult{
			OK: false, Mode: "draft_daemon_smoke", Tag: tag,
			ProxyChannels: ch, ProxySpatial: sp, Ready: ready, FirstBench: first, Error: err.Error(),
		}
		return out, err
	}

	second, err := sess.sendCommand(ctx, map[string]any{"cmd": "bench", "iters": iters})
	if err != nil {
		out := ANEDraftDaemonSmokeResult{
			OK: false, Mode: "draft_daemon_smoke", Tag: tag,
			ProxyChannels: ch, ProxySpatial: sp, Ready: ready,
			FirstBench: first, SecondBench: second, Error: err.Error(),
		}
		return out, err
	}

	handoff, handoffErr := ProbeANEMetalHandoffDims(ctx, ch, sp, quick)

	kernelReused := ready.CompileCount == 1 &&
		first.CompileCount == 1 &&
		second.CompileCount == 1 &&
		second.EvalCount == first.EvalCount+iters

	out := ANEDraftDaemonSmokeResult{
		OK:             handoffErr == nil && kernelReused,
		Mode:           "draft_daemon_smoke",
		Tag:            tag,
		ProxyChannels:  ch,
		ProxySpatial:   sp,
		Ready:          ready,
		FirstBench:     first,
		SecondBench:    second,
		OneShotHandoff: handoff,
		KernelReused:   kernelReused,
		Note:           "compile-once daemon session — ggml parent can hold surface_id across draft steps",
	}
	if handoffErr != nil {
		out.OK = false
		out.Error = handoffErr.Error()
		return out, handoffErr
	}
	if !kernelReused {
		out.OK = false
		out.Error = "kernel reuse check failed (compile_count or eval_count)"
		return out, fmt.Errorf("%s", out.Error)
	}
	return out, nil
}

// RunANEDraftDaemonSmokeJSON writes draft daemon smoke JSON to w.
func RunANEDraftDaemonSmokeJSON(ctx context.Context, w io.Writer, preferred string, quick bool) error {
	res, err := ProbeANEDraftDaemonSmoke(ctx, preferred, quick)
	enc := json.NewEncoder(w)
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}
