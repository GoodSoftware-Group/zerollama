package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// ErrHostMemNoReuse is returned when RAM/swap pressure blocks a load that cannot
// reuse an already-resident runner (cold model, dead process, or a different runner type).
var ErrHostMemNoReuse = errors.New("host RAM/swap exhausted; refusing a new model load")

type hostMemReuseLoadedKey struct{}

func withHostMemReuseLoaded(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, hostMemReuseLoadedKey{}, true)
}

func hostMemReuseLoaded(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	v, _ := ctx.Value(hostMemReuseLoadedKey{}).(bool)
	return v
}

func currentHostMemPressure() discover.HostMemPressure {
	return discover.HostMemSnapshot(
		envconfig.HostMemPressureRatio(),
		envconfig.HostSwapPressureRatio(),
		envconfig.HostSwapPressureFloor(),
	)
}

func (s *Server) hostMemGuard() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !envconfig.HostMemGuardEnabled() {
			c.Next()
			return
		}
		p := currentHostMemPressure()
		if !p.Pressure {
			c.Next()
			return
		}
		// Cold loads grow RSS. A resident model may still be served: scheduleRunner
		// pins memory-affecting options (num_ctx, KV type, draft) to the loaded
		// runner so a larger requested context truncates instead of reallocating.
		if model := hostMemGuardRequestModel(c); model != "" && s.modelResident(model) {
			slog.Info("host RAM/swap pressure; reusing loaded runner without growing KV",
				"model", model,
				"anon", p.Cgroup.Anon,
				"limit", p.Cgroup.Limit,
				"swap", p.Cgroup.SwapCurrent,
			)
			c.Request = c.Request.WithContext(withHostMemReuseLoaded(c.Request.Context()))
			c.Next()
			return
		}
		slog.Warn("refusing inference: host RAM/swap pressure",
			"anon", p.Cgroup.Anon,
			"limit", p.Cgroup.Limit,
			"swap", p.Cgroup.SwapCurrent,
		)
		writeBusyUnavailable(c, p.ClientMessage())
		c.Abort()
	}
}

func hostMemGuardRequestModel(c *gin.Context) string {
	if v, ok := c.Get(inferenceAccessLogKey); ok {
		if meta, ok := v.(*inferenceAccessMeta); ok && meta != nil {
			return strings.TrimSpace(meta.model)
		}
	}
	model, _, _ := inferencePeekRequest(c)
	return strings.TrimSpace(model)
}

// modelResident reports whether model is already in the ggml/MLX loaded set (/api/ps).
func (s *Server) modelResident(model string) bool {
	if s == nil || s.sched == nil || strings.TrimSpace(model) == "" {
		return false
	}
	return modelAlreadyWarm(s.sched.ProcessSnapshot(), model, nil)
}

// loadedReuse is a lock-free snapshot of one resident runner.
type loadedReuse struct {
	found         bool
	loading       bool
	dead          bool
	isImagegen    bool
	adaptersMatch bool
	contextShift  bool
	effectiveCtx  int
	runner        api.Runner
}

// pinRequestToLoadedRunner forces opts to the resident runner's memory shape.
// Sampling fields on opts are kept. A smaller request context is raised to the
// already-reserved KV; a larger one is capped so llama.cpp does not reload.
func pinRequestToLoadedRunner(opts api.Options, snap loadedReuse, wantImagegen bool) (api.Options, error) {
	if !snap.found {
		return opts, fmt.Errorf("%w (model is not loaded)", ErrHostMemNoReuse)
	}
	if snap.dead {
		return opts, fmt.Errorf("%w (loaded runner has exited; reload would allocate)", ErrHostMemNoReuse)
	}
	if snap.loading {
		// In-flight load already reserved its KV. needsReload reuses it as-is.
		return opts, nil
	}
	if snap.isImagegen != wantImagegen {
		return opts, fmt.Errorf("%w (loaded runner type does not match this request)", ErrHostMemNoReuse)
	}
	if !snap.adaptersMatch {
		return opts, fmt.Errorf("%w (adapters or projectors changed since load)", ErrHostMemNoReuse)
	}
	requested := opts.NumCtx
	opts.Runner = snap.runner
	if snap.effectiveCtx > 0 {
		opts.NumCtx = snap.effectiveCtx
	}
	if requested > opts.NumCtx && opts.NumCtx > 0 {
		slog.Warn("host RAM pressure: capping num_ctx to loaded KV",
			"requested", requested,
			"loaded", opts.NumCtx,
		)
	}
	return opts, nil
}

func hostMemoryStatusAPI() *api.HostMemoryStatus {
	p := currentHostMemPressure()
	st := &api.HostMemoryStatus{
		Pressure:         p.Pressure,
		Guard:            envconfig.HostMemGuardEnabled(),
		LimitBytes:       p.Cgroup.Limit,
		CurrentBytes:     p.Cgroup.Current,
		AnonBytes:        p.Cgroup.Anon,
		SwapCurrentBytes: p.Cgroup.SwapCurrent,
		Reason:           p.Reason,
		VideoAdmit: &api.VideoHostAdmitStatus{
			Ltx2BMinBytes:  uint64(ltx2bMinHostGiB) * format.GibiByte,
			Ltx13BMinBytes: uint64(ltxDefaultMinHostGiB) * format.GibiByte,
			Ltx25MinBytes:  uint64(ltx25MinHostGiB) * format.GibiByte,
			Unit:           "MemAvailable",
		},
	}
	if p.Cgroup.HasSwapMax {
		st.SwapMaxBytes = p.Cgroup.SwapMax
	}
	if mem, err := wanReadHostMem(); err == nil {
		st.AvailableBytes = mem.FreeMemory
		st.TotalBytes = mem.TotalMemory
	}
	return st
}
